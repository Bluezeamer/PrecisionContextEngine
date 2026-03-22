"""PCE 索引构建模块。

负责构建三层索引:
1. structure.md — 目录职责与模块清单
2. references.json — 符号引用索引 (通过 save_index)
3. annotations/index.md + annotations/modules/*.md — 渐进式项目认知导航

构建流程:
  list_dir 扫描目录
    → 并发 get_symbols_overview 建立符号清单
    → find_referencing_symbols 建立引用图
    → LLM 生成认知导航与模块认知文档(可降级)
    → 写入 .pce/annotations/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from ._env import build_litellm_model, get_completion_overrides, get_env_text
from .file_discovery import (
    HARD_SKIP_DIRS,
    SYMBOL_INDEX_EXTENSIONS,
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
    ProjectMeta,
    ReferenceEdge,
    ReferenceRelation,
    SymbolKind,
    SymbolRef,
)
from .module_registry import ModuleRegistryManager
from .serena_client import SerenaClient, SerenaClientError

logger = logging.getLogger(__name__)

# ============================================================================
# 常量配置
# ============================================================================

CODE_EXTENSIONS = SYMBOL_INDEX_EXTENSIONS
SKIP_DIRS = HARD_SKIP_DIRS

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
HIGH_LEVEL_KINDS = frozenset(
    {
        SymbolKind.CLASS,
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.MODULE,
    }
)

DEFAULT_CONCURRENCY = 10

ANNOTATIONS_DIR = "annotations"
ANNOTATIONS_INDEX_FILE = "index.md"
ANNOTATIONS_MODULES_DIR = "modules"


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


def _is_code_file(path: Path) -> bool:
    """判断路径是否为支持的代码文件。"""
    return supports_symbol_index(path)


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

    lines.extend(
        [
            "",
            "## 顶层模块清单",
            "| 模块 | 文件 | 符号数 |",
            "|------|------|--------|",
        ]
    )
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
# .pceignore 自动生成
# ============================================================================

_PCEIGNORE_MAX_DEPTH = 4
_PCEIGNORE_MAX_ITEMS = 240
_PCEIGNORE_FALLBACK_DIR_NAMES = {
    "temp",
    "tmp",
    "logs",
    "log",
    "artifacts",
    "exports",
    "output",
    "outputs",
    "coverage",
    "reports",
    "screenshots",
    "snapshots",
    ".cache",
    "cache",
}
_PCEIGNORE_FALLBACK_FILE_NAMES = {
    ".ds_store",
    "thumbs.db",
}


def _read_project_gitignore_excerpt(project_root: Path, *, max_lines: int = 80) -> str:
    path = project_root / ".gitignore"
    if not path.exists():
        return "(项目根无 .gitignore)"
    try:
        lines = path.read_text("utf-8").splitlines()
    except Exception:
        return "(项目根 .gitignore 读取失败)"
    kept = list(lines[:max_lines])
    if len(lines) > max_lines:
        kept.append(f"... (省略其余 {len(lines) - max_lines} 行)")
    return "\n".join(kept).strip() or "(项目根 .gitignore 为空)"


def _render_path_tree(paths: list[str], *, max_depth: int, max_items: int) -> str:
    normalized = _dedupe_keep_order(sorted(paths))[:max_items]
    lines: list[str] = []
    for raw_path in normalized:
        path = Path(raw_path)
        parts = path.parts[:max_depth]
        prefix = "  " * max(0, len(parts) - 1)
        label = parts[-1] if parts else raw_path
        suffix = "/" if len(path.parts) > 1 and len(parts) < len(path.parts) else ""
        lines.append(f"{prefix}- {label}{suffix}")
    if len(paths) > max_items:
        lines.append(f"... (省略其余 {len(paths) - max_items} 项)")
    return "\n".join(lines)


def _collect_pceignore_candidates(payload: Any) -> list[str]:
    dirs = []
    files = []
    if isinstance(payload, dict):
        raw_dirs = payload.get("dirs") or []
        raw_files = payload.get("files") or payload.get("items") or []
        dirs = [str(item) for item in raw_dirs if isinstance(item, (str, Path))]
        files = [str(item) for item in raw_files if isinstance(item, (str, Path))]

    candidates: list[str] = []
    for raw_dir in dirs:
        path = Path(raw_dir)
        if path.name.lower() in _PCEIGNORE_FALLBACK_DIR_NAMES:
            candidates.append(f"{path.as_posix().rstrip('/')}/")

    saw_log = False
    for raw_file in files:
        path = Path(raw_file)
        name = path.name.lower()
        if name in _PCEIGNORE_FALLBACK_FILE_NAMES:
            candidates.append(path.as_posix())
        if path.suffix.lower() == ".log":
            saw_log = True
    if saw_log:
        candidates.append("*.log")

    return _dedupe_keep_order(candidates)


def _build_pceignore_prompt(project_root: Path, payload: Any) -> str:
    dirs = []
    files = []
    if isinstance(payload, dict):
        raw_dirs = payload.get("dirs") or []
        raw_files = payload.get("files") or payload.get("items") or []
        dirs = [str(item) for item in raw_dirs if isinstance(item, (str, Path))]
        files = [str(item) for item in raw_files if isinstance(item, (str, Path))]

    tree = _render_path_tree(
        [*dirs, *files],
        max_depth=_PCEIGNORE_MAX_DEPTH,
        max_items=_PCEIGNORE_MAX_ITEMS,
    )
    gitignore_excerpt = _read_project_gitignore_excerpt(project_root)
    candidates = _collect_pceignore_candidates(payload)
    candidate_lines = [f"- {item}" for item in candidates] if candidates else ["(无候选规则)"]

    return "\n".join(
        [
            "请为 PCE 生成一个补充性的 `.pce/pceignore` 黑名单。",
            "目标：减少明显无价值、无关、冗余或高噪声文件进入索引/认知系统，但必须非常保守。",
            "",
            "已知约束：",
            "- 项目根 `.gitignore` 已优先生效；你生成的是补充规则，不是替代规则。",
            "- PCE 内建规则已跳过 `.git`、`.pce`、`.serena`、`node_modules`、`dist`、`build`、`__pycache__`、虚拟环境目录。",
            "- 不要忽略显然属于源码、配置、脚本、测试、迁移、文档设计稿中的关键说明文件，除非它们明显是生成产物或纯噪声。",
            "- 不要为了特定技术栈做假设；要使用尽量通用、保守的黑名单模式。",
            "- 优先输出目录模式或明确的产物文件模式，不要大面积使用扩展名通配导致误伤源码。",
            "",
            "你应该优先考虑忽略这些类别，但只有在目录树中确实出现时才输出：",
            "- 大型生成产物目录、缓存目录、导出目录、临时目录",
            "- 二进制资源、截图、打包产物、日志、锁文件（仅当你判断它们对代码理解价值很低）",
            "- 明显的本地私有配置或密钥类文件",
            "",
            "项目根 .gitignore 摘要：",
            gitignore_excerpt,
            "",
            "目录树摘要：",
            tree or "(空)",
            "",
            "允许选择的候选规则（只能从这里选，不能自创）：",
            *candidate_lines,
            "",
            "输出要求：",
            "- 只输出 gitignore 风格的模式，每行一条",
            "- 只能输出候选规则中的原文，不得改写、泛化或新增规则",
            "- 不要输出 Markdown、代码块、解释或编号",
            "- 如果没有明确需要补充忽略的模式，则输出空字符串",
        ]
    )


def _parse_pceignore_patterns(raw: str) -> list[str]:
    content = _strip_markdown_fence(raw)
    patterns: list[str] = []
    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("```"):
            continue
        text = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", text).strip()
        if not text or " " in text and not any(ch in text for ch in ("*", "/", ".", "?", "!")):
            continue
        if text.startswith("#"):
            continue
        patterns.append(text)
    return _dedupe_keep_order(patterns)


def _build_fallback_pceignore_patterns(payload: Any) -> list[str]:
    """在 LLM 不可用时，做极保守的补充黑名单推断。"""
    return _collect_pceignore_candidates(payload)


def _filter_safe_pceignore_patterns(patterns: list[str], payload: Any) -> list[str]:
    candidates = set(_collect_pceignore_candidates(payload))
    return [pattern for pattern in patterns if pattern in candidates]


async def _ensure_generated_pceignore(
    project_root: Path,
    serena_client: SerenaClient,
    *,
    model: str | None = None,
) -> None:
    path = project_root / ".pce" / "pceignore"
    if path.exists():
        return

    try:
        raw = await serena_client.list_dir(".", recursive=True, skip_ignored_files=True)
    except SerenaClientError as e:
        logger.warning("生成 .pce/pceignore 失败（目录扫描失败，已忽略）: %s", e)
        return

    payload = _normalize_tool_result(raw)
    prompt = _build_pceignore_prompt(project_root, payload)
    content = await _llm_complete_text(
        prompt,
        system_prompt=(
            "你是一个非常保守的代码索引黑名单生成器。"
            "你的职责是补充 gitignore 风格的排除模式，只排除明显的噪声或无价值路径，"
            "绝不能误伤潜在源码或关键工程文件。"
        ),
        model=model,
        failure_log=".pce/pceignore 生成失败",
    )
    patterns = _filter_safe_pceignore_patterns(
        _parse_pceignore_patterns(content) if content else [],
        payload,
    )
    if not patterns:
        patterns = _build_fallback_pceignore_patterns(payload)
    if not patterns:
        return

    final = "\n".join(
        [
            "# Auto-generated by PCE",
            "# Conservative supplemental ignore rules for indexing",
            *patterns,
            "",
        ]
    )
    await _atomic_write_text(path, final)
    logger.info("已生成 .pce/pceignore: %d 条规则", len(patterns))


# ============================================================================
# 渐进式认知导航生成
# ============================================================================


def _annotation_index_path(root_path: Path) -> Path:
    """返回 .pce/annotations/index.md 路径。"""
    return root_path / ".pce" / ANNOTATIONS_DIR / ANNOTATIONS_INDEX_FILE


def _annotation_modules_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/modules/ 目录路径。"""
    return root_path / ".pce" / ANNOTATIONS_DIR / ANNOTATIONS_MODULES_DIR


def _module_slug(module_name: str) -> str:
    """将模块名规范化为文件名 slug。"""
    normalized = " ".join(module_name.strip().split()) or "unnamed-module"
    return normalized.lower().replace(" ", "-")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """按出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _strip_markdown_fence(content: str) -> str:
    """去掉模型偶发输出的 Markdown 代码块包裹。"""
    raw = content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    return raw


def _extract_llm_text(response: Any) -> str:
    """从 litellm 响应中提取文本内容。"""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return getattr(msg, "content", "") or ""
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ""


def _resolve_annotation_model(model: str | None) -> str | None:
    """解析注解生成所用的模型配置，失败时返回 None（触发降级）。"""
    if model is not None:
        return model
    provider = get_env_text("PCE_PROVIDER")
    model_name = get_env_text("PCE_MODEL")
    if not provider or not model_name:
        logger.warning("未配置 PCE_PROVIDER 或 PCE_MODEL，跳过 LLM 认知文档生成")
        return None
    return build_litellm_model(provider, model_name)


async def _llm_complete_text(
    prompt: str,
    *,
    system_prompt: str,
    model: str | None,
    failure_log: str,
) -> str | None:
    """执行单次文本补全，返回去除代码块包裹后的纯文本，失败时返回 None。"""
    effective_model = _resolve_annotation_model(model)
    if effective_model is None:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=effective_model,
            messages=messages,
            temperature=0.1,
            **get_completion_overrides(),
        )
    except Exception as e:
        logger.warning(f"{failure_log}(已降级): {e}")
        return None

    content = _strip_markdown_fence(_extract_llm_text(response))
    return content.strip() or None



def _format_entry_outline(entry: IndexEntry) -> str:
    """将单个 IndexEntry 格式化为提示词摘要行（含主要符号名）。"""
    symbol_summary = ", ".join(
        f"{sym.name}({sym.kind.value})" for sym in entry.symbols[:8]
    ) or "无显式符号"
    if len(entry.symbols) > 8:
        symbol_summary += ", ..."
    return (
        f"- {entry.file_meta.path} "
        f"[{entry.file_meta.language}, {entry.file_meta.loc} 行, {len(entry.symbols)} 个符号] "
        f"符号: {symbol_summary}"
    )


def _build_index_md_prompt(entries: list[IndexEntry], project_meta: ProjectMeta) -> str:
    """构建生成 annotations/index.md 的 LLM 提示词（含 in-context learning 示例）。"""
    summary_lines = [_format_entry_outline(entry) for entry in entries[:80]]
    return "\n".join(
        [
            "你要为代码库生成一个可渐进加载的项目认知导航首页 index.md。",
            "请严格模仿示例的结构输出，不要输出任何额外解释，不要使用代码块。",
            "",
            "示例输出:",
            "# 项目认知导航",
            "",
            "## Agent Core",
            "文件：pce/agent.py, pce/agent_runtime/contracts.py, pce/agent_runtime/spawner.py",
            "职责：负责 PCE 主循环、任务编排与子 Agent 协议。对外接收查询目标，对内协调工具调用、compact 与交付流程。高风险点：ReAct 循环无 tool_calls 时的纠错逻辑、spawn 预算与深度限制。",
            "详细认知：.pce/annotations/modules/agent-core.md",
            "",
            "## Index Pipeline",
            "文件：pce/indexer.py, pce/memory.py",
            "职责：负责代码索引构建与持久化。维护结构索引、引用索引和渐进式认知文档；提供增量更新路径避免全量重建。高风险点：LLM 注解降级时需保证回退内容可机器解析。",
            "详细认知：.pce/annotations/modules/index-pipeline.md",
            "",
            f"项目根路径: {project_meta.root_path}",
            f"文件总数: {project_meta.file_count}, 代码行总数: {project_meta.loc_total}",
            "",
            "索引摘要:",
            *summary_lines,
            "",
            "输出约束:",
            "- 第一行必须是 `# 项目认知导航`",
            "- 每个模块使用 `## 模块名` 开头",
            "- `文件：` 行使用逗号分隔的相对路径，路径必须来自索引摘要",
            "- `职责：` 2-3 句话，描述模块边界、关键职责、主要协作对象和高风险点",
            "- `详细认知：` 写成 `.pce/annotations/modules/{slug}.md`，slug = 模块名小写后空格替换为连字符",
            "- 不要输出 JSON、代码块、前言、结语或任何未在示例中出现的附加文本",
        ]
    )


def _build_module_annotation_prompt(
    module_name: str,
    module_entries: list[IndexEntry],
) -> str:
    """构建单模块深度认知文档的 LLM 提示词。"""
    file_lines: list[str] = []
    for entry in module_entries:
        symbol_lines = ", ".join(
            f"{sym.name}({sym.kind.value})[{sym.line_start}-{sym.line_end}]"
            for sym in entry.symbols[:12]
        ) or "无显式符号"
        if len(entry.symbols) > 12:
            symbol_lines += ", ..."
        file_lines.append(
            f"- {entry.file_meta.path} [{entry.file_meta.language}, {entry.file_meta.loc} 行] "
            f"符号: {symbol_lines}"
        )

    return "\n".join(
        [
            "你要为单个代码模块生成可按需加载的深度认知文档。",
            "请输出 Markdown，不要代码块，不要额外解释。",
            "",
            f"模块名: {module_name}",
            f"覆盖文件数: {len(module_entries)}",
            "",
            "文件与符号摘要:",
            *file_lines,
            "",
            "输出要求:",
            f"- 标题必须是 `# {module_name}`",
            "- 必须包含 `## 覆盖文件`、`## 核心职责`、`## 关键符号`、`## 关键流程`、`## 外部协作`、`## 风险与约束` 这些二级标题",
            "- `## 覆盖文件` 需列出所有文件路径",
            "- `## 关键符号` 需优先引用给定摘要里的符号名和文件路径",
            "- `## 关键流程` 关注控制流、数据流或调用链，不要泛泛而谈",
            "- 结论必须基于输入，不得编造不存在的文件、符号或依赖",
        ]
    )


def _build_module_assignment_prompt(entry: IndexEntry, index_content: str) -> str:
    """构建新增文件模块归属判断的 LLM 提示词。"""
    return "\n".join(
        [
            "请根据现有项目认知导航，为一个新增文件判断最合适的模块归属。",
            "只能从已有模块 slug 中选择；如果无法可靠判断，则返回空字符串。",
            "",
            "现有导航:",
            index_content.strip(),
            "",
            "新增文件摘要:",
            _format_entry_outline(entry),
            "",
            "输出要求:",
            '- 只输出 JSON，例如 {"module_slug":"agent-core"}',
            '- 如果无法判断，输出 {"module_slug":""}',
            "- 不要输出 Markdown、代码块或解释",
        ]
    )


def _build_fallback_index_md(entries: list[IndexEntry]) -> str:
    """在 LLM 不可用时构建可机器解析的回退版 index.md（按顶层目录分组）。"""
    grouped: dict[str, list[IndexEntry]] = {}
    for entry in entries:
        path = Path(entry.file_meta.path)
        key = path.parts[0] if len(path.parts) > 1 else "__root__"
        grouped.setdefault(key, []).append(entry)

    lines = ["# 项目认知导航"]
    for key in sorted(grouped.keys()):
        module_entries = sorted(grouped[key], key=lambda e: str(e.file_meta.path))
        # 直接用 key 做 slug，避免来回转换引入噪声
        slug = key.lower().replace("_", "-").replace(" ", "-")
        module_name = key.replace("_", " ").replace("-", " ").title() if key != "__root__" else "Root Files"
        files = ", ".join(str(e.file_meta.path) for e in module_entries)
        lines.extend(
            [
                "",
                f"## {module_name}",
                f"文件：{files}",
                "职责：当前为基于目录结构的回退导航，建议模型可用时重新生成精细化认知边界。",
                f"详细认知：.pce/annotations/modules/{slug}.md",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_fallback_module_md(module_name: str, module_entries: list[IndexEntry]) -> str:
    """在 LLM 不可用时构建模块文档回退内容（基于符号索引）。"""
    lines = [
        f"# {module_name}",
        "",
        "## 覆盖文件",
    ]
    for entry in module_entries:
        lines.append(f"- {entry.file_meta.path}")

    lines.extend(["", "## 核心职责", "当前为基于索引条目的回退摘要，提供基本文件边界与符号概览。", "", "## 关键符号"])
    has_symbols = False
    for entry in module_entries:
        for sym in entry.symbols[:12]:
            lines.append(f"- {sym.name} ({sym.kind.value}) — {sym.file_path}:{sym.line_start}")
            has_symbols = True
    if not has_symbols:
        lines.append("- 无显式符号")

    lines.extend(
        [
            "",
            "## 关键流程",
            "需结合具体文件内容进一步检索，当前回退摘要不直接推断控制流。",
            "",
            "## 外部协作",
            "可结合 structure.md、references.json 及 Serena 工具继续定位模块交互关系。",
            "",
            "## 风险与约束",
            "本文件由索引级回退逻辑生成，细粒度职责与边界仍应以 LLM 生成版本为准。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# index.md 解析与增量更新辅助
# ============================================================================


def _parse_index_section(module_name: str, body_lines: list[str]) -> dict[str, Any]:
    """将 index.md 单个模块章节解析为结构化记录。"""
    file_paths: list[str] = []
    slug = _module_slug(module_name)
    for line in body_lines:
        m = re.match(r"^文件[:：]\s*(.+)$", line.strip())
        if m:
            file_paths = [p.strip() for p in m.group(1).split(",") if p.strip()]
            continue
        detail_match = re.search(r"modules/([^/\s]+)\.md", line.strip())
        if detail_match:
            slug = detail_match.group(1).strip()
    return {
        "name": module_name,
        "slug": slug,
        "body_lines": body_lines[:],
        "file_paths": file_paths,
    }


def _split_index_md(index_content: str) -> tuple[str, list[dict[str, Any]]]:
    """拆分 index.md 为头部文本和模块章节列表。"""
    header_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_name: str | None = None
    current_body: list[str] = []
    seen_section = False

    for raw_line in index_content.splitlines():
        if raw_line.startswith("## "):
            seen_section = True
            if current_name is not None:
                sections.append(_parse_index_section(current_name, current_body))
            current_name = raw_line[3:].strip()
            current_body = []
            continue
        if not seen_section:
            header_lines.append(raw_line)
        elif current_name is not None:
            current_body.append(raw_line)

    if current_name is not None:
        sections.append(_parse_index_section(current_name, current_body))

    header = "\n".join(header_lines).strip() or "# 项目认知导航"
    return header, sections


def _render_index_md(header: str, sections: list[dict[str, Any]]) -> str:
    """将结构化章节列表渲染回 index.md 字符串。"""
    lines = [header.strip() or "# 项目认知导航"]
    for section in sections:
        lines.extend(["", f"## {section['name']}"])
        lines.extend(section["body_lines"])
    return "\n".join(lines).rstrip() + "\n"


async def _cleanup_stale_module_docs(modules_dir: Path, section_slugs: set[str]) -> None:
    """删除未被当前 index sections 引用的模块文档。"""
    if not modules_dir.exists():
        return
    expected = {f"{slug}.md" for slug in section_slugs}
    for stale in modules_dir.glob("*.md"):
        if stale.name not in expected:
            await asyncio.to_thread(stale.unlink, missing_ok=True)


def _parse_index_md(index_content: str) -> dict[str, list[str]]:
    """返回模块 slug -> 文件路径列表的映射（供增量更新反查）。"""
    _, sections = _split_index_md(index_content)
    return {
        section["slug"]: _dedupe_keep_order(section["file_paths"])
        for section in sections
        if section["file_paths"]
    }


def _update_section_file_list(section: dict[str, Any], file_paths: list[str]) -> None:
    """就地替换章节中的 `文件：` 行，并更新 file_paths 字段。"""
    normalized = _dedupe_keep_order(file_paths)
    file_line = f"文件：{', '.join(normalized)}"
    new_body: list[str] = []
    replaced = False
    for line in section["body_lines"]:
        if not replaced and re.match(r"^文件[:：]\s*", line.strip()):
            new_body.append(file_line)
            replaced = True
        else:
            new_body.append(line)
    if not replaced:
        new_body.insert(0, file_line)
    section["body_lines"] = new_body
    section["file_paths"] = normalized


def _update_section_module_link(section: dict[str, Any], *, slug: str) -> None:
    """就地替换章节中的 `详细认知：` 行，并更新 slug。"""
    detail_line = f"详细认知：.pce/annotations/modules/{slug}.md"
    new_body: list[str] = []
    replaced = False
    for line in section["body_lines"]:
        if not replaced and re.match(r"^详细认知[:：]\s*", line.strip()):
            new_body.append(detail_line)
            replaced = True
        else:
            new_body.append(line)
    if not replaced:
        new_body.append(detail_line)
    section["body_lines"] = new_body
    section["slug"] = slug


def _fallback_slug_from_file(file_path: str) -> str:
    stem = Path(file_path).stem.lower().replace("_", "-").replace(".", "-")
    stem = re.sub(r"[^a-z0-9-]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")
    return stem or "unnamed-module"


_SHELL_FILENAMES = {
    "__init__.py",
    "main.py",
    "app.py",
    "models.py",
    "config.py",
    "settings.py",
}


def _build_temporary_section(file_path: str) -> dict[str, Any]:
    """为无法自动归属的新增文件创建临时模块章节。"""
    stem = Path(file_path).stem.replace(".", "-")
    module_name = f"临时归类 {stem}"
    slug = _fallback_slug_from_file(file_path)
    return {
        "name": module_name,
        "slug": slug,
        "file_paths": [file_path],
        "body_lines": [
            f"文件：{file_path}",
            "职责：该章节由增量索引降级逻辑创建，用于暂存未完成归类的新文件。建议后续执行全量索引校正模块边界。",
            f"详细认知：.pce/annotations/modules/{slug}.md",
        ],
    }


async def _load_sections_from_existing_module_docs(root_path: Path) -> list[dict[str, Any]]:
    """从已有 modules/*.md 中恢复章节信息。

    用途：
    - 当 LLM 生成的 index.md 漏掉部分现有模块时，避免直接退化为粗粒度 fallback 模块
    - 将已经存在的细粒度认知文档重新纳入 section 体系，后续再由 registry 稳定化
    """
    modules_dir = _annotation_modules_dir(root_path)
    if not modules_dir.exists():
        return []

    sections: list[dict[str, Any]] = []
    for module_path in sorted(modules_dir.glob("*.md")):
        try:
            raw = module_path.read_text("utf-8")
        except Exception:
            logger.warning(f"读取已有模块文档失败，跳过恢复: {module_path}")
            continue
        section = _parse_existing_module_doc_to_section(raw, module_path.stem)
        if section is not None:
            sections.append(section)
    return sections


def _parse_existing_module_doc_to_section(raw: str, slug: str) -> dict[str, Any] | None:
    lines = raw.splitlines()
    title = ""
    file_paths: list[str] = []
    in_cover_files = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not title:
            title = stripped[2:].strip()
            continue
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
            in_cover_files = heading == "覆盖文件"
            continue
        if in_cover_files and stripped.startswith("- "):
            payload = stripped[2:].strip()
            if payload:
                file_paths.append(payload)

    file_paths = _dedupe_keep_order(file_paths)
    if not title or not file_paths:
        return None

    return {
        "name": title,
        "slug": slug,
        "file_paths": file_paths,
        "body_lines": [
            f"文件：{', '.join(file_paths)}",
            f"详细认知：.pce/annotations/modules/{slug}.md",
        ],
    }


def _is_shell_like_file(file_path: str) -> bool:
    return Path(file_path).name in _SHELL_FILENAMES


def _fallback_group_key_for_file(file_path: str) -> str:
    path = Path(file_path)
    parent = path.parent.as_posix()
    if _is_shell_like_file(file_path):
        return f"shell::{parent if parent not in {'', '.'} else '__root__'}"
    return parent if parent not in {"", "."} else path.stem


def _fallback_display_name_from_group(group_key: str) -> str:
    if group_key.startswith("shell::"):
        raw = group_key.split("::", 1)[1]
        if raw == "__root__":
            return "补充归类 项目入口与共享模型"
        return f"补充归类 包级壳层 {raw}"
    label = group_key.replace("/", " / ")
    return f"补充归类 {label}"


def _build_fallback_section(file_paths: list[str]) -> dict[str, Any]:
    """为当前未归属的现存文件创建稳定 fallback 模块章节。"""
    normalized = _dedupe_keep_order(file_paths)
    group_key = _fallback_group_key_for_file(normalized[0]) if normalized else "misc"
    module_name = _fallback_display_name_from_group(group_key)
    slug = ModuleRegistryManager.normalize_slug(module_name, file_paths=normalized)
    return {
        "name": module_name,
        "slug": slug,
        "file_paths": normalized,
        "body_lines": [
            f"文件：{', '.join(normalized)}",
            "职责：该章节由索引阶段的 fallback 归类逻辑维护，用于承接尚未被主认知导航稳定吸纳的现存文件，避免这些文件长期游离于模块体系之外。",
            f"详细认知：.pce/annotations/modules/{slug}.md",
        ],
    }


def _is_fallback_section(section: dict[str, Any]) -> bool:
    name = str(section.get("name") or "")
    return name.startswith("补充归类 ")


def _append_fallback_sections_for_unassigned_entries(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
) -> list[dict[str, Any]]:
    """为当前未命中任何章节的现存文件追加稳定 fallback 模块。

    设计目的：
    - 避免文件已经进入 index / baseline，但长期没有 module registry 归属
    - 让后续 deleted path 也能通过当前模块覆盖获得更稳定的历史映射
    """
    assigned = {
        file_path
        for section in sections
        for file_path in _dedupe_keep_order(section.get("file_paths", []))
        if file_path in entries_map
    }
    unassigned = sorted(path for path in entries_map if path not in assigned)
    if not unassigned:
        return sections

    grouped: dict[str, list[str]] = {}
    for file_path in unassigned:
        grouped.setdefault(_fallback_group_key_for_file(file_path), []).append(file_path)

    augmented = list(sections)
    sections_by_slug = {section["slug"]: section for section in augmented}
    for file_paths in grouped.values():
        fallback = _build_fallback_section(file_paths)
        existing = sections_by_slug.get(fallback["slug"])
        if existing is not None:
            _update_section_file_list(
                existing,
                existing.get("file_paths", []) + fallback["file_paths"],
            )
            continue
        augmented.append(fallback)
        sections_by_slug[fallback["slug"]] = fallback
    return augmented


def _merge_missing_sections(
    base_sections: list[dict[str, Any]],
    extra_sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将缺失 section 合并进来，并优先让细粒度现有模块拆分 fallback 覆盖。"""
    merged = list(base_sections)
    known_slugs = {section["slug"] for section in merged}

    def _covered_paths(*, include_fallback: bool) -> set[str]:
        return {
            file_path
            for section in merged
            if include_fallback or not _is_fallback_section(section)
            for file_path in section.get("file_paths", [])
        }

    for section in extra_sections:
        if section["slug"] in known_slugs:
            continue
        extra_files = _dedupe_keep_order(section.get("file_paths", []))
        covered_without_fallback = _covered_paths(include_fallback=False)
        if not any(path not in covered_without_fallback for path in extra_files):
            continue

        for existing in merged:
            if not _is_fallback_section(existing):
                continue
            overlap = [
                path
                for path in existing.get("file_paths", [])
                if path in extra_files
            ]
            if not overlap:
                continue
            remaining = [
                path
                for path in existing.get("file_paths", [])
                if path not in set(extra_files)
            ]
            _update_section_file_list(existing, remaining)

        merged.append(section)
        known_slugs.add(section["slug"])
    return [section for section in merged if section.get("file_paths")]


def _dedupe_sections_by_slug(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 slug 去重，优先保留正式 section，并合并文件列表。"""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for section in sections:
        slug = section["slug"]
        if slug not in merged:
            merged[slug] = section
            order.append(slug)
            continue

        current = merged[slug]
        prefer = current
        fallback = section
        if _is_fallback_section(current) and not _is_fallback_section(section):
            prefer = section
            fallback = current

        _update_section_file_list(
            prefer,
            prefer.get("file_paths", []) + fallback.get("file_paths", []),
        )
        merged[slug] = prefer

    return [merged[slug] for slug in order]


def _prepare_module_specs(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, list[IndexEntry]]]]:
    """过滤章节中实际存在的文件，返回有效章节列表和模块生成规格。"""
    valid_sections: list[dict[str, Any]] = []
    module_specs: list[tuple[str, str, list[IndexEntry]]] = []
    for section in sections:
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            logger.warning(f"认知导航章节未命中实际文件，跳过: {section['name']}")
            continue
        _update_section_file_list(section, file_paths)
        valid_sections.append(section)
        module_specs.append(
            (section["name"], section["slug"], [entries_map[p] for p in file_paths])
        )
    return valid_sections, module_specs


def _extract_module_key_symbols(module_entries: list[IndexEntry]) -> list[str]:
    symbols: list[str] = []
    for entry in module_entries:
        symbols.extend(sym.name for sym in entry.symbols[:12])
    return _dedupe_keep_order(symbols)


async def _stabilize_sections_with_registry(
    sections: list[dict[str, Any]],
    *,
    root_path: Path,
    entries_map: dict[str, IndexEntry],
) -> list[dict[str, Any]]:
    """根据 module registry 稳定模块 slug。"""
    manager = ModuleRegistryManager(root_path)
    stabilized: list[dict[str, Any]] = []
    for section in sections:
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            stabilized.append(section)
            continue
        record = await manager.get_or_create_module(
            display_name=section["name"],
            file_paths=file_paths,
            key_symbols=_extract_module_key_symbols([entries_map[p] for p in file_paths]),
        )
        section["name"] = record.display_name
        _update_section_file_list(section, file_paths)
        _update_section_module_link(section, slug=record.slug)
        stabilized.append(section)
    return stabilized


async def _generate_index_md(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    model: str | None = None,
) -> str | None:
    """调用 LLM 生成 annotations/index.md。"""
    if not entries:
        return None
    return await _llm_complete_text(
        _build_index_md_prompt(entries, project_meta),
        system_prompt="你是资深软件架构分析助手，擅长把大型代码库整理成可渐进加载的认知导航。",
        model=model,
        failure_log="认知导航 index.md 生成失败",
    )


async def _generate_module_annotation(
    module_name: str,
    module_entries: list[IndexEntry],
    model: str | None = None,
) -> str | None:
    """调用 LLM 生成单模块深度认知文档。"""
    if not module_entries:
        return None
    return await _llm_complete_text(
        _build_module_annotation_prompt(module_name, module_entries),
        system_prompt="你是资深软件架构分析助手，擅长生成可按需加载的模块深度认知文档。",
        model=model,
        failure_log=f"模块认知文档生成失败: {module_name}",
    )


async def _classify_new_entry_module(
    entry: IndexEntry,
    index_content: str,
    known_slugs: set[str],
    model: str | None = None,
) -> str | None:
    """用 LLM 判断新增文件应归属的模块 slug，不在 known_slugs 中时返回 None。"""
    if not known_slugs:
        return None
    content = await _llm_complete_text(
        _build_module_assignment_prompt(entry, index_content),
        system_prompt="你是代码库模块归类助手，只能从现有模块中选择最匹配的归属。",
        model=model,
        failure_log=f"新增文件模块归属判断失败: {entry.file_meta.path}",
    )
    if not content:
        return None
    slug = ""
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            slug = str(payload.get("module_slug") or "").strip()
    except json.JSONDecodeError:
        slug = content.strip()
    return slug if slug in known_slugs else None


async def _write_module_files(
    module_specs: list[tuple[str, str, list[IndexEntry]]],
    modules_dir: Path,
    model: str | None,
) -> None:
    """并发生成并写入 modules/*.md，失败时使用回退内容。"""
    results = await asyncio.gather(
        *[_generate_module_annotation(name, entries, model=model) for name, _, entries in module_specs],
        return_exceptions=True,
    )
    for (module_name, slug, module_entries), result in zip(module_specs, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(f"模块认知文档生成异常，使用回退内容: {module_name}: {result}")
            content = None
        else:
            content = result
        if not content:
            content = _build_fallback_module_md(module_name, module_entries)
        await _atomic_write_text(modules_dir / f"{slug}.md", content.rstrip() + "\n")


async def _write_annotations(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    model: str | None = None,
) -> None:
    """全量生成并写入 annotations/index.md 与 modules/*.md。"""
    index_path = _annotation_index_path(root_path)
    modules_dir = _annotation_modules_dir(root_path)

    if not entries:
        await _atomic_write_text(index_path, "# 项目认知导航\n")
        if modules_dir.exists():
            for stale in modules_dir.glob("*.md"):
                await asyncio.to_thread(stale.unlink, missing_ok=True)
        return

    index_content = await _generate_index_md(entries, project_meta, model=model)
    if not index_content:
        logger.warning("认知导航 index.md 不可用，使用索引级回退模板")
        index_content = _build_fallback_index_md(entries)

    entries_map = {str(e.file_meta.path): e for e in entries}
    header, sections = _split_index_md(index_content)
    sections = _dedupe_sections_by_slug(sections)
    sections = _merge_missing_sections(
        sections,
        await _load_sections_from_existing_module_docs(root_path),
    )
    sections = _append_fallback_sections_for_unassigned_entries(sections, entries_map)
    sections = await _stabilize_sections_with_registry(
        sections,
        root_path=root_path,
        entries_map=entries_map,
    )
    valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    if not valid_sections:
        logger.warning("认知导航解析为空，回退到结构化模板")
        index_content = _build_fallback_index_md(entries)
        header, sections = _split_index_md(index_content)
        valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    await _write_module_files(module_specs, modules_dir, model)
    rendered_index = _render_index_md(header, valid_sections)
    await _atomic_write_text(index_path, rendered_index)
    await _cleanup_stale_module_docs(
        modules_dir,
        {section["slug"] for section in valid_sections},
    )
    logger.info(
        f"认知导航写入完成: {len(valid_sections)} 个模块章节, {len(module_specs)} 个模块文档"
    )


async def _update_annotations_incremental(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    *,
    changed_files: list[str],
    deleted_files: list[str],
    model: str | None = None,
) -> None:
    """增量更新受影响模块的认知文档与 index.md。"""
    index_path = _annotation_index_path(root_path)
    modules_dir = _annotation_modules_dir(root_path)

    try:
        async with aiofiles.open(index_path, encoding="utf-8") as f:
            index_content = await f.read()
    except FileNotFoundError:
        logger.info("未找到认知导航 index.md，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return
    except Exception as e:
        logger.warning(f"读取认知导航失败，降级为全量认知文档重建: {e}")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    header, sections = _split_index_md(index_content)
    if not sections:
        logger.warning("认知导航解析结果为空，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    entries_map = {str(e.file_meta.path): e for e in entries}
    sections = _dedupe_sections_by_slug(sections)
    sections = _merge_missing_sections(
        sections,
        await _load_sections_from_existing_module_docs(root_path),
    )
    ownership = {
        section["slug"]: _dedupe_keep_order(section["file_paths"])
        for section in sections
        if section["file_paths"]
    }
    reverse_map: dict[str, str] = {
        file_path: slug
        for slug, file_paths in ownership.items()
        for file_path in file_paths
    }
    sections_by_slug = {section["slug"]: section for section in sections}
    affected_slugs: set[str] = set()

    # 已知文件的变更/删除 → 标记所属模块为受影响
    for file_path in (*changed_files, *deleted_files):
        slug = reverse_map.get(file_path)
        if slug:
            affected_slugs.add(slug)

    # 新增文件（不在任何已知模块中）→ 尝试归属或创建临时章节
    for file_path in changed_files:
        if file_path in reverse_map:
            continue
        entry = entries_map.get(file_path)
        if entry is None:
            continue

        current_index_content = _render_index_md(header, sections)
        assigned_slug = await _classify_new_entry_module(
            entry, current_index_content, set(sections_by_slug.keys()), model=model
        )
        if assigned_slug and assigned_slug in sections_by_slug:
            section = sections_by_slug[assigned_slug]
            _update_section_file_list(section, section["file_paths"] + [file_path])
            affected_slugs.add(assigned_slug)
            logger.info(f"新增文件已归属既有模块: {file_path} -> {assigned_slug}")
        else:
            temp_section = _build_temporary_section(file_path)
            sections.append(temp_section)
            sections_by_slug[temp_section["slug"]] = temp_section
            affected_slugs.add(temp_section["slug"])
            logger.warning(f"新增文件归属判断失败，已追加临时章节: {file_path}")

    sections = _append_fallback_sections_for_unassigned_entries(sections, entries_map)
    sections_by_slug = {section["slug"]: section for section in sections}

    if not affected_slugs:
        logger.info("认知导航无受影响模块，跳过 annotations 增量更新")
        return

    # 重新生成受影响模块；无文件的模块移除
    final_sections: list[dict[str, Any]] = []
    removed_slugs: set[str] = set()

    for section in sections:
        slug = section["slug"]
        if slug not in affected_slugs:
            final_sections.append(section)
            continue
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            removed_slugs.add(slug)
            logger.info(f"模块已无文件，移除认知章节: {section['name']}")
            continue
        _update_section_file_list(section, file_paths)
        final_sections.append(section)

    final_sections = await _stabilize_sections_with_registry(
        final_sections,
        root_path=root_path,
        entries_map=entries_map,
    )
    final_sections, module_specs = _prepare_module_specs(final_sections, entries_map)

    await _write_module_files(module_specs, modules_dir, model)
    rendered_index = _render_index_md(header, final_sections)
    await _atomic_write_text(index_path, rendered_index)
    await _cleanup_stale_module_docs(
        modules_dir,
        {section["slug"] for section in final_sections},
    )
    logger.info(
        f"认知导航增量更新完成: {len(module_specs)} 个模块重建, {len(removed_slugs)} 个模块移除"
    )


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

    symbols: list[SymbolRef] = []
    edges: list[ReferenceEdge] = []
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

            # 对高层符号建立引用索引
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

    try:
        await _ensure_generated_pceignore(root_path, serena_client, model=model)
    except Exception as e:
        logger.warning(f"生成 .pce/pceignore 失败（已忽略）: {e}")

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
    results = await asyncio.gather(*[_run_with_semaphore(f) for f in files], return_exceptions=True)

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
    await save_index(snapshot, root_path=memory_root_path)
    await _write_structure_md(entries, memory_root_path)

    logger.info(
        f"索引构建完成: {len(entries)} 个文件, "
        f"{build_stats.total_symbols} 个符号, "
        f"{build_stats.total_edges} 条引用边, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # 渐进式认知导航(可降级)
    try:
        await _write_annotations(entries, project_meta, memory_root_path, model=model)
    except Exception as e:
        logger.warning(f"写入项目认知导航失败(已降级): {e}")

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
        await _ensure_generated_pceignore(root_path, serena_client, model=model)
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
        logger.info("无有效变更，跳过增量更新")
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
    await _write_structure_md(merged_entries, memory_root_path)

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
        )
    except Exception as e:
        logger.warning(f"增量更新项目认知导航失败(已降级): {e}")

    return snapshot
