"""PCE 索引构建模块。

负责文件发现和符号索引构建。认知文档生成逻辑已迁出到 annotation_writer.py。

构建流程:
  list_dir 扫描目录
    → 并发 get_symbols_overview 建立文件/符号快照
    → 调用 annotation_writer 生成认知导航文档
    → 写入 .pce/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from .file_discovery import (
    HARD_SKIP_DIRS,
    filter_trackable_files,
    is_hard_skipped,
    should_track_deleted_path,
    should_track_existing_file,
    supports_symbol_index,
)
from .memory import load_index, save_index
from .models import (
    BuildStats,
    FileMeta,
    IndexEntry,
    IndexSnapshot,
    NavigationTree,
    ProjectMeta,
    SymbolKind,
    SymbolRef,
)
from .serena_client import SerenaClient, SerenaClientError
from .serena_language_registry import infer_language_for_path

logger = logging.getLogger(__name__)

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
    return is_hard_skipped(path)


def _infer_language(path: Path) -> str:
    """从文件扩展名推断语言标识。"""
    return infer_language_for_path(path) or (path.suffix.lstrip(".") or "text")


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
        "class",
        "interface",
        "method",
        "function",
        "module",
        "file",
        "variable",
        "import",
        "enum",
        "property",
        "constructor",
        "field",
        "constant",
        "namespace",
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


_STRUCTURE_STATE_FILE = "structure_state.json"
_STRUCTURE_REQUIRED_HEADINGS: tuple[str, ...] = (
    "## 项目形态概览",
    "## 顶层区域",
    "## 关键入口候选",
    "## 模块对齐提示",
    "## 导航建议",
)
_STRUCTURE_HEDGE_MARKERS: tuple[str, ...] = (
    "候选",
    "可优先",
    "可先",
    "可视为",
    "可能",
    "线索",
    "提示",
    "建议",
)
_STRUCTURE_CAUTIOUS_SECTION_HEADINGS: tuple[str, ...] = (
    "## 项目形态概览",
    "## 顶层区域",
    "## 关键入口候选",
    "## 模块对齐提示",
)
_FRONTEND_HINTS = frozenset({"frontend", "web", "ui", "client", "vite", "react", "vue"})
_BACKEND_HINTS = frozenset({"backend", "api", "server", "service", "fastapi", "flask"})
_SCRIPT_HINTS = frozenset({"script", "scripts", "tool", "tools", "cli", "cmd", "bin"})
_TEST_HINTS = frozenset({"test", "tests", "spec", "specs"})
_DOC_HINTS = frozenset({"doc", "docs", "design", "designs"})
_CONFIG_HINTS = frozenset({"config", "configs", "setting", "settings", "env"})
_ENTRY_FILE_PRIORITY: dict[str, tuple[int, str]] = {
    "main.py": (100, "文件名命中 `main.*` 规则"),
    "main.ts": (100, "文件名命中 `main.*` 规则"),
    "main.js": (100, "文件名命中 `main.*` 规则"),
    "main.rs": (100, "文件名命中 `main.*` 规则"),
    "app.py": (96, "文件名命中 `app.*` 规则"),
    "app.ts": (96, "文件名命中 `app.*` 规则"),
    "app.js": (96, "文件名命中 `app.*` 规则"),
    "server.py": (94, "文件名命中 `server.*` 规则"),
    "server.ts": (94, "文件名命中 `server.*` 规则"),
    "server.js": (94, "文件名命中 `server.*` 规则"),
    "cli.py": (92, "文件名命中 `cli.*` 规则"),
    "cli.ts": (92, "文件名命中 `cli.*` 规则"),
    "cli.js": (92, "文件名命中 `cli.*` 规则"),
    "__main__.py": (90, "文件名命中 Python 包入口规则"),
    "lib.rs": (88, "文件名命中 Rust 库入口规则"),
    "index.ts": (84, "文件名命中 `index.*` 规则"),
    "index.js": (84, "文件名命中 `index.*` 规则"),
}
_DIRECTORY_HINT_GROUPS: tuple[tuple[frozenset[str], str], ...] = (
    (_FRONTEND_HINTS, "路径 token 命中前端相关词"),
    (_BACKEND_HINTS, "路径 token 命中服务端相关词"),
    (_SCRIPT_HINTS, "路径 token 命中脚本/工具相关词"),
    (_TEST_HINTS, "路径 token 命中测试相关词"),
    (_DOC_HINTS, "路径 token 命中文档相关词"),
    (_CONFIG_HINTS, "路径 token 命中配置相关词"),
)
_MISSING_COVERAGE_REPAIR_BATCH_SIZE = 24
_MISSING_FACT_FULL_LINES = 60
_MISSING_FACT_WINDOW_LINES = 20
_MISSING_KIND_VALUES = {
    "implementation",
    "config",
    "documentation",
    "test",
    "resource",
    "shell",
    "entrypoint",
    "unknown",
}



def _annotation_refresh_reason(root_path: Path) -> str | None:
    """判断现有 annotations 是否需要在无代码变更时强制刷新。"""
    annotations_dir = root_path / ".pce" / "annotations"
    tree_path = annotations_dir / "navigation_tree.json"
    index_path = annotations_dir / "index.md"
    areas_dir = annotations_dir / "areas"

    if not tree_path.exists():
        return "缺少 navigation_tree.json"
    if not index_path.exists():
        return "缺少 index.md"
    if not areas_dir.exists():
        return "缺少 areas/ 目录"

    try:
        payload = json.loads(tree_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"navigation_tree.json 读取失败: {exc}"

    expected_version = str(NavigationTree.model_fields["version"].default)
    actual_version = str(payload.get("version") or "").strip()
    if actual_version != expected_version:
        return (
            f"navigation_tree 版本过期: current={actual_version or '(missing)'} "
            f"expected={expected_version}"
        )

    areas = payload.get("areas")
    if not isinstance(areas, list) or not areas:
        return "navigation_tree 缺少 areas"

    try:
        index_content = index_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"index.md 读取失败: {exc}"
    if "## 区域入口" not in index_content:
        return "index.md 缺少区域入口章节"

    area_files = list(areas_dir.glob("*.md"))
    if len(area_files) < len(areas):
        return "areas 文档数量不足"

    return None


# ============================================================================
# 认知文档生成（已迁出到 annotation_writer.py）
# ============================================================================

from .annotation_writer import (  # noqa: E402
    _write_annotations,
    _update_annotations_incremental,
    _write_structure_md,
)
from .pceignore_stage import run_pceignore_stage


# ============================================================================
# 核心索引逻辑
# ============================================================================

async def _scan_directory(serena_client: SerenaClient) -> list[str]:
    """递归扫描项目目录,返回应纳入 PCE 的文本文件列表。"""
    try:
        raw = await serena_client.list_dir(".", recursive=True, skip_ignored_files=True)
    except SerenaClientError as e:
        logger.error(f"目录扫描失败: {e}")
        return []

    payload = _normalize_tool_result(raw)
    files = _extract_file_list(payload)
    results = filter_trackable_files(serena_client.project_path, files)

    logger.info(f"扫描完成: 发现 {len(results)} 个可跟踪文本文件")
    return results


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

    symbols: list[SymbolRef] = []
    if supports_symbol_index(file_path):
        try:
            overview_raw = await serena_client.get_symbols_overview(file_path, depth=1)
        except SerenaClientError as e:
            logger.info(f"符号概览不可用，降级为空符号索引: {file_path}: {e}")
        else:
            overview = _normalize_tool_result(overview_raw)
            for sym_dict in _flatten_symbols(overview):
                sym = _symbol_from_dict(sym_dict, file_path)
                if sym is not None:
                    symbols.append(sym)

    # 收集文件统计信息
    try:
        stat = abs_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        size_bytes = stat.st_size
    except Exception:
        mtime = datetime.now(UTC)
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
        edges=[],
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

    pceignore_start = time.monotonic()
    try:
        await run_pceignore_stage(root_path, serena_client, model=model)
    except Exception as e:
        logger.warning(f"生成 .pce/pceignore 失败（已忽略）: {e}")
    finally:
        logger.info("索引阶段耗时: pceignore=%.2fs", time.monotonic() - pceignore_start)

    # 扫描文件列表
    scan_start = time.monotonic()
    files = await _scan_directory(serena_client)
    logger.info(
        "索引阶段耗时: scan_directory=%.2fs (files=%d)",
        time.monotonic() - scan_start,
        len(files),
    )
    if not files:
        logger.warning("未发现任何源代码文件")

    # 并发索引所有文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    # return_exceptions=True 确保单文件异常不会中断整体构建
    file_index_start = time.monotonic()
    results = await asyncio.gather(*[_run_with_semaphore(f) for f in files], return_exceptions=True)
    logger.info(
        "索引阶段耗时: file_indexing=%.2fs",
        time.monotonic() - file_index_start,
    )

    entries: list[IndexEntry] = []
    failed_files: list[str] = []
    for file_path, result in zip(files, results, strict=False):
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
    created_at = datetime.now(UTC)
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
    save_index_start = time.monotonic()
    await save_index(snapshot, root_path=memory_root_path)
    logger.info(
        "索引阶段耗时: save_index=%.2fs",
        time.monotonic() - save_index_start,
    )

    logger.info(
        f"索引构建完成: {len(entries)} 个文件, "
        f"{build_stats.total_symbols} 个符号, "
        f"{build_stats.total_edges} 条预构建引用边, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # 渐进式认知导航(可降级)
    annotations_start = time.monotonic()
    try:
        await _write_annotations(
            entries,
            project_meta,
            memory_root_path,
            model=model,
            serena_client=serena_client,
        )
    except Exception as e:
        logger.warning(f"写入项目认知导航失败(已降级): {e}")
    finally:
        logger.info(
            "索引阶段耗时: write_annotations=%.2fs",
            time.monotonic() - annotations_start,
        )

    structure_start = time.monotonic()
    try:
        await _write_structure_md(
            entries,
            memory_root_path,
            model=model,
            force_refresh=True,
        )
    except Exception as e:
        logger.warning(f"写入 structure.md 失败(已降级): {e}")
    finally:
        logger.info(
            "索引阶段耗时: write_structure=%.2fs",
            time.monotonic() - structure_start,
        )

    return snapshot


async def build_index_incremental(
    project_path: str | Path,
    serena_client: SerenaClient,
    memory_root: str | Path | None = None,
    *,
    changed_files: list[str],
    deleted_files: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str | None = None,
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

    try:
        await run_pceignore_stage(root_path, serena_client, model=model)
    except Exception as e:
        logger.warning(f"生成 .pce/pceignore 失败（已忽略）: {e}")

    # 尝试加载已有索引
    existing = await load_index(root_path=memory_root_path)
    if existing is None:
        logger.info("无历史索引，降级为全量构建")
        return await build_index(
            project_path=project_path,
            serena_client=serena_client,
            memory_root=memory_root,
            concurrency=concurrency,
            model=model,
        )

    # ignore-first：增量时只过滤掉被忽略/非文本的现存文件；
    # 删除文件只排除内建硬规则目录，避免遗留旧 entry / baseline。
    effective_changes = [
        f for f in changed_files if should_track_existing_file(root_path, f)
    ]
    deleted = {path for path in (deleted_files or []) if should_track_deleted_path(path)}

    if not effective_changes and not deleted:
        refresh_reason = _annotation_refresh_reason(memory_root_path)
        if refresh_reason is None:
            logger.info("无有效变更，跳过增量更新")
            return existing

        logger.info("无有效变更，但 annotations 需要刷新: %s", refresh_reason)
        await _write_annotations(
            existing.entries,
            existing.project_meta,
            memory_root_path,
            model=model,
            serena_client=serena_client,
        )
        try:
            await _write_structure_md(
                existing.entries,
                memory_root_path,
                model=model,
                force_refresh=False,
            )
        except Exception as e:
            logger.warning(f"刷新 structure.md 失败(已忽略): {e}")
        return existing

    logger.info(f"开始增量索引: {len(effective_changes)} 个变更, " f"{len(deleted)} 个删除")

    # 并发重建变更文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    results = await asyncio.gather(*[_run(f) for f in effective_changes], return_exceptions=True)

    # 以现有条目为基础，按文件路径建立映射
    entries_map: dict[str, IndexEntry] = {str(e.file_meta.path): e for e in existing.entries}

    failed_files: list[str] = []
    for file_path, result in zip(effective_changes, results, strict=False):
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
        created_at=datetime.now(UTC),
        build_stats=build_stats,
    )

    # 写入 Memory
    await save_index(snapshot, root_path=memory_root_path)

    logger.info(
        f"增量索引完成: {len(effective_changes)} 文件更新, "
        f"{len(deleted)} 文件删除, "
        f"合计 {len(merged_entries)} 文件, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # 增量更新认知导航(可降级)
    try:
        await _update_annotations_incremental(
            merged_entries,
            project_meta,
            memory_root_path,
            changed_files=effective_changes,
            deleted_files=sorted(deleted),
            model=model,
            serena_client=serena_client,
        )
    except Exception as e:
        logger.warning(f"增量更新项目认知导航失败(已降级): {e}")

    try:
        await _write_structure_md(
            merged_entries,
            memory_root_path,
            model=model,
            force_refresh=False,
        )
    except Exception as e:
        logger.warning(f"增量更新 structure.md 失败(已降级): {e}")

    return snapshot
