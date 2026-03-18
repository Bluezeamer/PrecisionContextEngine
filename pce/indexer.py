"""PCE 索引构建模块。

负责构建三层索引:
1. structure.md — 目录职责与模块清单
2. references.json — 符号引用索引 (通过 save_index)
3. annotations.md — 语义注解 (通过 append_memory)

构建流程:
  list_dir 扫描目录
    → 并发 get_symbols_overview 建立符号清单
    → find_referencing_symbols 建立引用图
    → LLM 生成语义注解(可降级)
    → 写入 Memory
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from .memory import append_memory, load_index, save_index
from .models import (
    BuildStats,
    FileMeta,
    IndexEntry,
    IndexSnapshot,
    MemoryItem,
    MemoryItemType,
    ProjectMeta,
    ReferenceEdge,
    ReferenceRelation,
    SymbolKind,
    SymbolRef,
)
from .serena_client import SerenaClient, SerenaClientError
from ._env import get_completion_overrides, get_env_text, normalize_litellm_model

logger = logging.getLogger(__name__)

# ============================================================================
# 常量配置
# ============================================================================

CODE_EXTENSIONS = frozenset({
    ".py", ".ts", ".js", ".tsx", ".jsx",
    ".go", ".java", ".rs", ".cpp", ".c", ".h",
})

SKIP_DIRS = frozenset({
    ".git", "node_modules", "dist", "build",
    "__pycache__", ".venv", "venv", ".pce",
})

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
}

# 只对这些符号类型构建引用索引(避免过多 LLM 调用)
HIGH_LEVEL_KINDS = frozenset({
    SymbolKind.CLASS,
    SymbolKind.FUNCTION,
    SymbolKind.METHOD,
    SymbolKind.MODULE,
})

DEFAULT_CONCURRENCY = 10


# ============================================================================
# 工具函数
# ============================================================================


def _normalize_tool_result(value: Any) -> Any:
    """将工具返回值统一化为 dict/list 结构。

    Serena 工具返回值经过 _jsonable 处理后可能有多种形态：
    1. 直接返回 dict/list（最理想）
    2. 单元素列表包含 JSON 字符串（旧版兼容）
    3. {'meta': ..., 'content': [{'type': 'text', 'text': <data>}]} 外壳（实测形态）
    """
    # 处理 content 外壳结构：{'meta': ..., 'content': [...]}
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            # 取第一个 type=text 的条目（兼容多元素 content）
            text_item = next(
                (item for item in content if isinstance(item, dict) and item.get("type") == "text"),
                None,
            )
            if text_item is not None:
                inner = text_item["text"]
                if isinstance(inner, str):
                    try:
                        return json.loads(inner)
                    except (ValueError, json.JSONDecodeError):
                        return inner
                return inner
    # 处理单元素列表包含 JSON 字符串的情况（旧版兼容）
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        try:
            return json.loads(value[0])
        except (ValueError, json.JSONDecodeError):
            return value[0]
    return value


def _should_skip(path: Path) -> bool:
    """判断路径是否在跳过列表中。"""
    return any(part in SKIP_DIRS for part in path.parts)


def _is_code_file(path: Path) -> bool:
    """判断路径是否为支持的代码文件。"""
    return path.suffix.lower() in CODE_EXTENSIONS


def _infer_language(path: Path) -> str:
    """从文件扩展名推断语言标识。"""
    return LANGUAGE_MAP.get(path.suffix.lower(), path.suffix.lstrip(".") or "text")


def _count_lines(path: Path) -> int:
    """统计文件行数,失败时返回 0。"""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        logger.debug(f"统计行数失败: {path}")
        return 0


def _extract_file_list(payload: Any) -> list[str]:
    """从 list_dir 响应中提取文件路径列表。"""
    if isinstance(payload, dict):
        files = payload.get("files") or payload.get("items") or []
        if isinstance(files, list):
            return [str(f) for f in files]
    if isinstance(payload, list):
        return [str(f) for f in payload if isinstance(f, (str, Path))]
    return []


def _flatten_symbols(payload: Any) -> list[dict[str, Any]]:
    """从 get_symbols_overview 响应中提取所有符号。

    Serena 的 get_symbols_overview 返回按类型分组的嵌套结构：
      {"Class": ["Foo", {"Bar": {"Method": ["baz"]}}], "Function": ["qux"]}
    键 = 符号类型, 值 = 列表, 元素为字符串(简单符号)或 dict(嵌套容器)。
    此函数将其展平为 [{"name": ..., "kind": ...}, ...] 的统一列表。
    """
    _KIND_KEYS = {
        "class", "interface", "method", "function", "module",
        "file", "variable", "import", "enum", "property",
        "constructor", "field", "constant", "namespace",
    }
    result: list[dict[str, Any]] = []

    def _is_kind_group(d: dict[str, Any]) -> bool:
        """判断 dict 的键是否全部为符号类型名。"""
        return bool(d) and all(str(k).lower() in _KIND_KEYS for k in d)

    def _walk(node: Any, kind: str | None) -> None:
        if isinstance(node, str):
            if kind:
                result.append({"name": node, "kind": kind})
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, kind)
            return
        if isinstance(node, dict):
            # 情况 1: {"Class": [...], "Function": [...]} — 按类型分组
            if _is_kind_group(node):
                for k, v in node.items():
                    _walk(v, str(k))
                return
            # 情况 2: {"SymbolRef": {"Method": [...]}} — 嵌套容器符号
            for name, children in node.items():
                if kind:
                    result.append({"name": str(name), "kind": kind})
                _walk(children, None)

    _walk(payload, None)
    return result


def _flatten_references(payload: Any) -> list[dict[str, Any]]:
    """从 find_referencing_symbols 响应中提取所有引用 dict。"""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        result: list[dict[str, Any]] = []
        for v in payload.values():
            result.extend(_flatten_references(v))
        return result
    return []


def _map_symbol_kind(raw: Any) -> SymbolKind:
    """将 Serena 的 kind 字段映射到 SymbolKind 枚举。"""
    if raw is None:
        return SymbolKind.FUNCTION
    key = str(raw).lower().replace("symbolkind.", "")
    mapping: dict[str, SymbolKind] = {
        "class": SymbolKind.CLASS,
        "interface": SymbolKind.CLASS,
        "method": SymbolKind.METHOD,
        "function": SymbolKind.FUNCTION,
        "module": SymbolKind.MODULE,
        "file": SymbolKind.MODULE,
        "variable": SymbolKind.VARIABLE,
        "import": SymbolKind.IMPORT,
    }
    return mapping.get(key, SymbolKind.FUNCTION)


def _symbol_from_dict(payload: dict[str, Any], file_path: str) -> SymbolRef | None:
    """从符号字典构建 SymbolRef 对象。"""
    name = payload.get("name") or payload.get("name_path")
    if not name:
        return None

    # 解析行号范围
    line_start = 1
    line_end = 1
    location = payload.get("body_location") or payload.get("location") or {}
    if isinstance(location, dict):
        start = location.get("start_line") or location.get("line")
        end = location.get("end_line")
        if isinstance(start, int) and start > 0:
            line_start = start
        if isinstance(end, int) and end >= line_start:
            line_end = end
        else:
            line_end = line_start

    try:
        return SymbolRef(
            symbol_id=str(uuid.uuid4()),
            name=str(name),
            kind=_map_symbol_kind(payload.get("kind")),
            file_path=Path(file_path),
            line_start=line_start,
            line_end=line_end,
            signature=None,
        )
    except Exception as e:
        logger.debug(f"构建 SymbolRef 失败: {name}: {e}")
        return None


def _edge_from_reference(target: SymbolRef, ref: dict[str, Any]) -> ReferenceEdge:
    """从引用字典构建 ReferenceEdge 对象。"""
    ref_path = ref.get("relative_path") or ""
    location = ref.get("body_location") or {}
    line_start = location.get("start_line")

    # 构建证据字符串
    parts: list[str] = []
    if ref_path:
        parts.append(ref_path)
        if isinstance(line_start, int) and line_start > 0:
            parts.append(f":{line_start}")
    snippet = ref.get("content_around_reference") or ref.get("snippet") or ""
    if snippet:
        parts.append(f" {snippet[:80]}")  # 截断过长的片段

    evidence = "".join(parts) or target.name

    return ReferenceEdge(
        from_symbol_id=target.symbol_id,
        to_symbol_id=str(uuid.uuid4()),
        relation=ReferenceRelation.USES,
        evidence=evidence,
    )


# ============================================================================
# 原子文件写入
# ============================================================================


async def _atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        await asyncio.to_thread(os.replace, tmp_path, path)
    except Exception:
        if tmp_path.exists():
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
        raise


# ============================================================================
# structure.md 生成
# ============================================================================


def _build_structure_md(entries: list[IndexEntry]) -> str:
    """生成 structure.md 文本内容。"""
    # 收集顶层目录
    top_dirs: dict[str, list[str]] = {}
    for entry in entries:
        path = Path(entry.file_meta.path)
        parts = path.parts
        top_dir = f"{parts[0]}/" if len(parts) > 1 else "./"
        top_dirs.setdefault(top_dir, []).append(path.as_posix())

    lines = ["# 项目结构索引\n", "## 目录职责"]
    for directory in sorted(top_dirs.keys()):
        file_count = len(top_dirs[directory])
        lines.append(f"- `{directory}` — {file_count} 个文件")

    lines.extend([
        "",
        "## 顶层模块清单",
        "| 模块 | 文件 | 符号数 |",
        "|------|------|--------|",
    ])
    for entry in sorted(entries, key=lambda e: str(e.file_meta.path)):
        path = Path(entry.file_meta.path)
        module = path.stem or path.name
        file_path = path.as_posix()
        symbol_count = len(entry.symbols)
        lines.append(f"| {module} | {file_path} | {symbol_count} |")

    return "\n".join(lines) + "\n"


async def _write_structure_md(entries: list[IndexEntry], root_path: Path) -> None:
    """将 structure.md 写入 .pce 目录。"""
    path = root_path / ".pce" / "structure.md"
    content = _build_structure_md(entries)
    await _atomic_write_text(path, content)


# ============================================================================
# LLM 语义注解生成
# ============================================================================


def _build_annotation_prompt(entries: list[IndexEntry], project_meta: ProjectMeta) -> str:
    """构建语义注解的 LLM 提示词。"""
    summary_lines = [
        f"- {entry.file_meta.path} "
        f"[{entry.file_meta.language}, {entry.file_meta.loc} 行, "
        f"{len(entry.symbols)} 个符号]"
        for entry in entries[:50]  # 限制提示词长度
    ]

    return "\n".join([
        "你是软件架构助手,请基于以下项目索引摘要生成模块职责语义注解。",
        "",
        f"项目根路径: {project_meta.root_path}",
        f"文件总数: {project_meta.file_count}, 代码行总数: {project_meta.loc_total}",
        "",
        "索引摘要:",
        *summary_lines,
        "",
        "输出要求(Markdown 格式):",
        "- 按模块/目录划分章节,使用 ## 标题",
        "- 每章包含: 核心职责、关键流程、依赖关系、高风险点",
        "- 语言简洁,每章不超过 200 字",
    ])


async def _generate_annotations(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    model: str | None = None,
) -> str | None:
    """调用 LLM 生成语义注解。失败时降级返回 None。"""
    if not entries:
        return None

    effective_model = (
        model
        or get_env_text("PCE_ANNOTATION_MODEL")
        or get_env_text("PCE_MODEL")
    )
    effective_model = normalize_litellm_model(effective_model)
    if not effective_model:
        logger.warning("未配置 PCE_ANNOTATION_MODEL 或 PCE_MODEL，跳过语义注解生成")
        return None

    prompt = _build_annotation_prompt(entries, project_meta)
    messages = [
        {"role": "system", "content": "你是软件架构助手,擅长总结模块职责与依赖关系。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=effective_model,
            messages=messages,
            temperature=0.2,
            **get_completion_overrides(),
        )
        # 提取响应文本
        content = ""
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", "") or ""
        elif isinstance(response, dict):
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() or None
    except Exception as e:
        logger.warning(f"语义注解生成失败(已降级): {e}")
        return None


# ============================================================================
# 核心索引逻辑
# ============================================================================


async def _scan_directory(serena_client: SerenaClient) -> list[str]:
    """递归扫描项目目录,返回源代码文件路径列表。"""
    try:
        raw = await serena_client.list_dir(".", recursive=True, skip_ignored_files=True)
    except SerenaClientError as e:
        logger.error(f"目录扫描失败: {e}")
        return []

    payload = _normalize_tool_result(raw)
    files = _extract_file_list(payload)

    results: list[str] = []
    for item in files:
        path = Path(item)
        if _should_skip(path) or not _is_code_file(path):
            continue
        results.append(path.as_posix())

    logger.info(f"扫描完成: 发现 {len(results)} 个源代码文件")
    return sorted(set(results))


async def _resolve_name_path(
    serena_client: SerenaClient, symbol: SymbolRef, file_path: str
) -> str | None:
    """解析符号的 name_path(Serena 格式,如 MyClass/my_method)。"""
    try:
        raw = await serena_client.find_symbol(
            symbol.name,
            relative_path=file_path,
            include_body=False,
            depth=0,
            substring_matching=False,
        )
    except SerenaClientError:
        return None

    payload = _normalize_tool_result(raw)
    for item in _flatten_symbols(payload):
        name_path = item.get("name_path") or item.get("namePath")
        if name_path:
            return str(name_path)
    return None


async def _index_file(file_path: str, serena_client: SerenaClient) -> IndexEntry | None:
    """索引单个文件,返回 IndexEntry。

    Args:
        file_path: 相对于项目根目录的文件路径
        serena_client: 已连接的 SerenaClient 实例

    Returns:
        索引条目,如果文件不可访问则返回 None
    """
    root_path = serena_client.project_path
    abs_path = root_path / file_path

    if not abs_path.exists():
        logger.warning(f"文件不存在,跳过: {file_path}")
        return None

    # 获取符号概览
    try:
        overview_raw = await serena_client.get_symbols_overview(file_path, depth=1)
    except SerenaClientError as e:
        logger.warning(f"获取符号概览失败: {file_path}: {e}")
        return None

    overview = _normalize_tool_result(overview_raw)
    symbols: list[SymbolRef] = []
    for sym_dict in _flatten_symbols(overview):
        sym = _symbol_from_dict(sym_dict, file_path)
        if sym is not None:
            symbols.append(sym)

    # 对高层符号建立引用索引
    edges: list[ReferenceEdge] = []
    for symbol in symbols:
        if symbol.kind not in HIGH_LEVEL_KINDS:
            continue

        name_path = await _resolve_name_path(serena_client, symbol, file_path)
        if not name_path:
            continue

        try:
            refs_raw = await serena_client.find_referencing_symbols(
                name_path=name_path, relative_path=file_path
            )
        except SerenaClientError:
            continue

        refs_payload = _normalize_tool_result(refs_raw)
        for ref in _flatten_references(refs_payload):
            edges.append(_edge_from_reference(symbol, ref))

    # 收集文件统计信息
    try:
        stat = abs_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        size_bytes = stat.st_size
    except Exception:
        mtime = datetime.now(timezone.utc)
        size_bytes = 0

    return IndexEntry(
        file_meta=FileMeta(
            path=Path(file_path),
            language=_infer_language(abs_path),
            size_bytes=size_bytes,
            mtime=mtime,
            loc=_count_lines(abs_path),
        ),
        symbols=symbols,
        imports=[],
        edges=edges,
    )


# ============================================================================
# 公开 API
# ============================================================================


async def build_index(
    project_path: str | Path,
    serena_client: SerenaClient,
    memory_root: str | Path | None = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str | None = None,
) -> IndexSnapshot:
    """构建三层索引并写入 Memory。

    Args:
        project_path: 目标项目根路径
        serena_client: 已连接的 Serena 客户端
        memory_root: Memory 文件的写入根路径,默认与 project_path 相同
        concurrency: 并发索引文件数上限
        model: 用于生成语义注解的 LLM 模型名称(litellm 格式)

    Returns:
        构建完成的 IndexSnapshot

    Raises:
        Exception: 索引构建失败
    """
    root_path = Path(project_path).resolve()
    memory_root_path = Path(memory_root).resolve() if memory_root else root_path
    start_time = time.monotonic()

    logger.info(f"开始构建索引: {root_path}")

    # 扫描文件列表
    files = await _scan_directory(serena_client)
    if not files:
        logger.warning("未发现任何源代码文件")

    # 并发索引所有文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    # return_exceptions=True 确保单文件异常不会中断整体构建
    results = await asyncio.gather(
        *[_run_with_semaphore(f) for f in files], return_exceptions=True
    )

    entries: list[IndexEntry] = []
    failed_files: list[str] = []
    for file_path, result in zip(files, results):
        if isinstance(result, BaseException):
            logger.warning(f"文件索引异常: {file_path}: {result}")
            failed_files.append(file_path)
        elif result is None:
            failed_files.append(file_path)
        else:
            entries.append(result)
    warnings = [f"索引失败: {path}" for path in failed_files]

    if warnings:
        logger.warning(f"有 {len(warnings)} 个文件索引失败")

    # 构建元数据
    created_at = datetime.now(timezone.utc)
    project_meta = ProjectMeta(
        root_path=root_path,
        created_at=created_at,
        index_version="1",
        file_count=len(entries),
        loc_total=sum(e.file_meta.loc for e in entries),
    )

    build_stats = BuildStats(
        total_files=len(entries),
        total_symbols=sum(len(e.symbols) for e in entries),
        total_edges=sum(len(e.edges) for e in entries),
        duration_ms=int((time.monotonic() - start_time) * 1000),
        warnings=warnings,
    )

    snapshot = IndexSnapshot(
        project_meta=project_meta,
        entries=entries,
        created_at=created_at,
        build_stats=build_stats,
    )

    # 写入 Memory
    await save_index(snapshot, root_path=memory_root_path)
    await _write_structure_md(entries, memory_root_path)

    logger.info(
        f"索引构建完成: {len(entries)} 个文件, "
        f"{build_stats.total_symbols} 个符号, "
        f"{build_stats.total_edges} 条引用边, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # LLM 语义注解(可降级)
    annotations = await _generate_annotations(entries, project_meta, model=model)
    if annotations:
        try:
            memory_item = MemoryItem(
                item_id=str(uuid.uuid4()),
                item_type=MemoryItemType.SUMMARY,
                content=annotations,
                tags=["annotations", "index"],
                related_files=[],
                created_at=datetime.now(timezone.utc),
            )
            await append_memory(memory_item, root_path=memory_root_path)
            logger.info("语义注解已写入 Memory")
        except Exception as e:
            logger.warning(f"写入语义注解失败(已降级): {e}")

    return snapshot


async def build_index_incremental(
    project_path: str | Path,
    serena_client: SerenaClient,
    memory_root: str | Path | None = None,
    *,
    changed_files: list[str],
    deleted_files: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> IndexSnapshot:
    """增量索引更新：仅重建变更文件，合并到已有索引。

    若不存在历史索引，自动降级为全量 build_index。

    Args:
        project_path: 目标项目根路径
        serena_client: 已连接的 Serena 客户端
        memory_root: Memory 文件的写入根路径
        changed_files: 变更（含新增）的文件相对路径列表
        deleted_files: 已删除的文件相对路径列表
        concurrency: 并发索引文件数上限

    Returns:
        合并后的 IndexSnapshot
    """
    root_path = Path(project_path).resolve()
    memory_root_path = Path(memory_root).resolve() if memory_root else root_path
    start_time = time.monotonic()

    # 尝试加载已有索引
    existing = await load_index(root_path=memory_root_path)
    if existing is None:
        logger.info("无历史索引，降级为全量构建")
        return await build_index(
            project_path=project_path,
            serena_client=serena_client,
            memory_root=memory_root,
            concurrency=concurrency,
        )

    # 过滤有效的代码文件
    effective_changes = [
        f for f in changed_files
        if _is_code_file(Path(f)) and not _should_skip(Path(f))
    ]
    deleted = set(deleted_files or [])

    if not effective_changes and not deleted:
        logger.info("无有效变更，跳过增量更新")
        return existing

    logger.info(
        f"开始增量索引: {len(effective_changes)} 个变更, "
        f"{len(deleted)} 个删除"
    )

    # 并发重建变更文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    results = await asyncio.gather(
        *[_run(f) for f in effective_changes], return_exceptions=True
    )

    # 以现有条目为基础，按文件路径建立映射
    entries_map: dict[str, IndexEntry] = {
        str(e.file_meta.path): e for e in existing.entries
    }

    failed_files: list[str] = []
    for file_path, result in zip(effective_changes, results):
        if isinstance(result, BaseException):
            logger.warning(f"增量索引异常: {file_path}: {result}")
            failed_files.append(file_path)
        elif result is None:
            failed_files.append(file_path)
        else:
            entries_map[str(result.file_meta.path)] = result

    # 移除已删除文件的条目
    for path in deleted:
        entries_map.pop(path, None)

    merged_entries = list(entries_map.values())
    warnings = [f"增量索引失败: {path}" for path in failed_files]

    # 更新元数据（保留原始 created_at 和 index_version）
    project_meta = ProjectMeta(
        root_path=root_path,
        created_at=existing.project_meta.created_at,
        index_version=existing.project_meta.index_version,
        file_count=len(merged_entries),
        loc_total=sum(e.file_meta.loc for e in merged_entries),
    )

    build_stats = BuildStats(
        total_files=len(merged_entries),
        total_symbols=sum(len(e.symbols) for e in merged_entries),
        total_edges=sum(len(e.edges) for e in merged_entries),
        duration_ms=int((time.monotonic() - start_time) * 1000),
        warnings=warnings,
    )

    snapshot = IndexSnapshot(
        project_meta=project_meta,
        entries=merged_entries,
        created_at=datetime.now(timezone.utc),
        build_stats=build_stats,
    )

    # 写入 Memory
    await save_index(snapshot, root_path=memory_root_path)
    await _write_structure_md(merged_entries, memory_root_path)

    logger.info(
        f"增量索引完成: {len(effective_changes)} 文件更新, "
        f"{len(deleted)} 文件删除, "
        f"合计 {len(merged_entries)} 文件, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    return snapshot
