"""PCE 索引构建模块。

负责构建三层索引:
1. structure.md — 项目结构导航
2. references.json — 符号引用索引 (通过 save_index)
3. annotations/index.md + annotations/modules/*.md — 渐进式项目认知导航

构建流程:
  list_dir 扫描目录
    → 并发 get_symbols_overview 建立文件/符号快照
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
import tomllib
import uuid
from collections import Counter, defaultdict
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
    ModuleRegistry,
    ProjectMeta,
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

DEFAULT_CONCURRENCY = 10

ANNOTATIONS_DIR = "annotations"
ANNOTATIONS_INDEX_FILE = "index.md"
ANNOTATIONS_AREAS_DIR = "areas"
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


def _structure_path(root_path: Path) -> Path:
    return root_path / ".pce" / "structure.md"


def _structure_state_path(root_path: Path) -> Path:
    return root_path / ".pce" / _STRUCTURE_STATE_FILE


def _tokenize_path_hint(raw: str) -> set[str]:
    return {part for part in re.split(r"[\/_.-]+", raw.lower()) if part}


def _compact_markdown_block(content: str) -> str:
    normalized: list[str] = []
    previous_blank = False
    for raw in content.splitlines():
        line = raw.rstrip()
        blank = not line.strip()
        if blank:
            if previous_blank or not normalized:
                continue
            previous_blank = True
            normalized.append("")
            continue
        previous_blank = False
        normalized.append(line)
    while normalized and not normalized[-1].strip():
        normalized.pop()
    return "\n".join(normalized).strip()


def _select_representative_paths(entries: list[IndexEntry], max_items: int = 3) -> list[str]:
    ranked = sorted(
        entries,
        key=lambda entry: (
            len(Path(entry.file_meta.path).parts),
            -len(entry.symbols),
            -entry.file_meta.loc,
            str(entry.file_meta.path),
        ),
    )
    return [str(entry.file_meta.path) for entry in ranked[:max_items]]


def _summarize_path_token_hints(tokens: set[str]) -> list[str]:
    hints: list[str] = []
    for candidates, label in _DIRECTORY_HINT_GROUPS:
        matched = sorted(tokens & candidates)
        if matched:
            rendered = "/".join(f"`{item}`" for item in matched[:4])
            hints.append(f"{label}（{rendered}）")
    return hints


def _format_language_summary(entries: list[IndexEntry], *, max_items: int = 4) -> str:
    counter = Counter(entry.file_meta.language for entry in entries)
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    preview = [f"{lang}({count})" for lang, count in items[:max_items]]
    if len(items) > max_items:
        preview.append(f"... (+{len(items) - max_items} 种)")
    return "、".join(preview) or "未知"


def _describe_project_shape_candidate(
    top_dir_entries: dict[str, list[IndexEntry]],
    entry_points: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized = sorted(directory for directory in top_dir_entries if directory != "./")
    token_union: set[str] = set()
    for directory in normalized:
        token_union.update(_tokenize_path_hint(directory))

    has_frontend = bool(token_union & _FRONTEND_HINTS)
    has_backend = bool(token_union & _BACKEND_HINTS)
    has_scripts = bool(token_union & _SCRIPT_HINTS)

    if has_frontend and has_backend and has_scripts:
        label = "前后端分层 + 工具链混合"
    elif has_frontend and has_backend:
        label = "前后端并存"
    elif len(normalized) <= 1:
        label = "单主目录"
    elif len(normalized) >= 4:
        label = "多目录混合"
    else:
        label = "多区域协作"

    evidence = [
        f"顶层目录 {len(normalized) + (1 if './' in top_dir_entries else 0)} 个",
        f"入口候选 {len(entry_points)} 个",
    ]
    token_hints = _summarize_path_token_hints(token_union)
    evidence.extend(token_hints[:3] if token_hints else ["未命中明显技术栈目录 token"])
    return {"label": label, "evidence": evidence}


def _build_top_level_candidates(top_dir_entries: dict[str, list[IndexEntry]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for directory, entries in sorted(top_dir_entries.items()):
        tokens = _tokenize_path_hint(directory)
        hints = _summarize_path_token_hints(tokens)
        evidence = [f"文件数 {len(entries)}", f"语言分布 {_format_language_summary(entries, max_items=3)}"]
        evidence.extend(hints[:2] if hints else ["未命中明显路径语义 token"])
        candidates.append(
            {
                "path": directory,
                "file_count": len(entries),
                "representatives": _select_representative_paths(entries, max_items=3),
                "evidence": evidence,
            }
        )
    return candidates


def _find_entry_points(entries: list[IndexEntry], *, max_items: int = 8) -> list[dict[str, Any]]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for entry in entries:
        path = Path(entry.file_meta.path)
        path_str = path.as_posix()
        name = path.name.lower()
        reason = ""
        role_info = _ENTRY_FILE_PRIORITY.get(name)
        if role_info is not None:
            priority, reason = role_info
        elif path.parent.name.lower() in {"bin", "cmd"}:
            priority, reason = 80, "父目录命中 `bin/cmd` 规则"
        elif name.startswith("test_"):
            priority, reason = 48, "文件名命中测试脚本规则"
        else:
            continue

        if "config" in name or name.endswith(".config.js") or name.endswith(".config.ts"):
            continue
        if path.parts and path.parts[0].lower() in {"docs", "doc"}:
            continue
        priority += max(0, 8 - len(path.parts))
        candidates.append(
            (
                priority,
                {
                    "path": path_str,
                    "evidence": [
                        reason,
                        f"路径深度 {len(path.parts)}",
                        f"语言 {entry.file_meta.language}",
                    ],
                },
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, payload in sorted(candidates, key=lambda item: (-item[0], item[1]["path"])):
        path_str = payload["path"]
        if path_str in seen:
            continue
        seen.add(path_str)
        deduped.append(payload)
        if len(deduped) >= max_items:
            break
    return deduped


def _select_core_areas(top_dir_entries: dict[str, list[IndexEntry]], *, max_items: int = 6) -> list[dict[str, Any]]:
    areas: list[dict[str, Any]] = []

    for directory, entries in top_dir_entries.items():
        if directory == "./":
            continue

        nested_groups: dict[str, list[IndexEntry]] = defaultdict(list)
        for entry in entries:
            parts = Path(entry.file_meta.path).parts
            if len(parts) >= 3:
                nested_groups["/".join(parts[:2]) + "/"].append(entry)

        selected_groups = [
            (prefix, group_entries)
            for prefix, group_entries in nested_groups.items()
            if len(group_entries) >= 2
        ]
        if not selected_groups:
            selected_groups = [(directory, entries)]

        ranked_groups = sorted(
            selected_groups,
            key=lambda item: (
                -len(item[1]),
                -sum(len(entry.symbols) for entry in item[1]),
                item[0],
            ),
        )
        for prefix, group_entries in ranked_groups[:2]:
            tokens = _tokenize_path_hint(prefix)
            evidence = [
                f"文件数 {len(group_entries)}",
                f"符号数 {sum(len(entry.symbols) for entry in group_entries)}",
                f"语言分布 {_format_language_summary(group_entries, max_items=3)}",
            ]
            evidence.extend(_summarize_path_token_hints(tokens)[:2])
            areas.append(
                {
                    "prefix": prefix,
                    "file_count": len(group_entries),
                    "symbol_count": sum(len(entry.symbols) for entry in group_entries),
                    "representatives": _select_representative_paths(group_entries, max_items=3),
                    "evidence": evidence,
                }
            )

    areas = sorted(
        areas,
        key=lambda item: (-item["file_count"], -item["symbol_count"], item["prefix"]),
    )
    return areas[:max_items]


def _build_module_alignment_hints(
    area_candidates: list[dict[str, Any]],
    index_sections: list[dict[str, Any]],
    *,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for area in area_candidates:
        prefix = area["prefix"]
        matched = [
            section["name"]
            for section in index_sections
            if any(file_path.startswith(prefix) for file_path in section.get("file_paths", []))
        ]
        if not matched:
            continue
        hints.append(
            {
                "prefix": prefix,
                "modules": _dedupe_keep_order(matched)[:4],
                "evidence": [f"匹配到 {len(_dedupe_keep_order(matched))} 个模块章节"],
            }
        )
        if len(hints) >= max_items:
            break
    return hints


def _build_navigation_tips(
    entry_points: list[dict[str, Any]],
    area_candidates: list[dict[str, Any]],
) -> list[str]:
    tips: list[str] = []
    if entry_points:
        tips.append(f"若先确认运行链路，可先查看 `{entry_points[0]['path']}`。")
    if len(entry_points) > 1:
        tips.append(f"若首个入口线索不足，可继续查看 `{entry_points[1]['path']}`。")
    if area_candidates:
        tips.append(
            "若先做结构定位，可优先浏览 "
            + "、".join(f"`{area['prefix']}`" for area in area_candidates[:3])
            + "。"
        )
    tips.append("若弱提示已足够，继续转到 `.pce/annotations/index.md` 按模块下钻。")
    return tips


def _build_structure_rule_bundle(
    entries: list[IndexEntry],
    *,
    index_sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    top_dir_entries: dict[str, list[IndexEntry]] = {}
    for entry in entries:
        path = Path(entry.file_meta.path)
        parts = path.parts
        top_dir = f"{parts[0]}/" if len(parts) > 1 else "./"
        top_dir_entries.setdefault(top_dir, []).append(entry)

    entry_points = _find_entry_points(entries)
    area_candidates = _select_core_areas(top_dir_entries)
    return {
        "project_shape": _describe_project_shape_candidate(top_dir_entries, entry_points),
        "top_level_candidates": _build_top_level_candidates(top_dir_entries),
        "entrypoint_candidates": entry_points,
        "area_candidates": area_candidates,
        "module_alignment_hints": _build_module_alignment_hints(
            area_candidates,
            index_sections or [],
        ),
        "navigation_tips": _build_navigation_tips(entry_points, area_candidates),
    }


def _render_rule_structure_md(bundle: dict[str, Any]) -> str:
    lines = ["# 项目结构导航", "", "## 项目形态候选"]
    shape = bundle["project_shape"]
    lines.append(f"- 候选：{shape['label']}")
    lines.extend(f"  - 依据：{item}" for item in shape["evidence"])

    lines.extend(["", "## 顶层区域候选"])
    for item in bundle["top_level_candidates"]:
        representatives = "、".join(f"`{path}`" for path in item["representatives"]) or "(无)"
        lines.append(f"- `{item['path']}`")
        lines.extend(f"  - 观测：{evidence}" for evidence in item["evidence"])
        lines.append(f"  - 代表：{representatives}")

    lines.extend(["", "## 入口点候选"])
    if bundle["entrypoint_candidates"]:
        for item in bundle["entrypoint_candidates"]:
            lines.append(f"- `{item['path']}`")
            lines.extend(f"  - 依据：{evidence}" for evidence in item["evidence"])
    else:
        lines.append("- 暂未识别到明显入口点候选。")

    lines.extend(["", "## 高密度代码区域候选"])
    if bundle["area_candidates"]:
        for item in bundle["area_candidates"]:
            representatives = "、".join(f"`{path}`" for path in item["representatives"]) or "(无)"
            lines.append(f"- `{item['prefix']}`")
            lines.extend(f"  - 观测：{evidence}" for evidence in item["evidence"])
            lines.append(f"  - 代表：{representatives}")
    else:
        lines.append("- 暂未识别到明显的高密度代码区域候选。")

    lines.extend(["", "## 模块对齐提示"])
    if bundle["module_alignment_hints"]:
        for item in bundle["module_alignment_hints"]:
            modules = "、".join(f"`{name}`" for name in item["modules"])
            lines.append(f"- `{item['prefix']}` ↔ {modules}")
            lines.extend(f"  - 依据：{evidence}" for evidence in item["evidence"])
    else:
        lines.append("- 暂无稳定的区域与模块对齐提示。")

    lines.extend(["", "## 导航建议"])
    lines.extend(f"- {tip}" for tip in bundle["navigation_tips"])
    return "\n".join(lines).rstrip() + "\n"


def _build_structure_refresh_signals(
    bundle: dict[str, Any],
    *,
    index_sections: list[dict[str, Any]],
    registry: ModuleRegistry,
) -> dict[str, Any]:
    active_records = [record for record in registry.records.values() if record.status == "active"]
    return {
        "project_shape": bundle["project_shape"]["label"],
        "top_level_paths": [item["path"] for item in bundle["top_level_candidates"]],
        "entrypoint_paths": [item["path"] for item in bundle["entrypoint_candidates"]],
        "area_prefixes": [item["prefix"] for item in bundle["area_candidates"]],
        "index_slugs": sorted(section["slug"] for section in index_sections),
        "fallback_sections": sum(
            1 for section in index_sections if section["name"].startswith("补充归类 ")
        ),
        "active_module_ids": sorted(record.module_id for record in active_records),
        "active_module_slugs": sorted({record.slug for record in active_records}),
    }


async def _load_structure_state(root_path: Path) -> dict[str, Any] | None:
    path = _structure_state_path(root_path)
    if not path.exists():
        return None
    try:
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


async def _save_structure_state(root_path: Path, signals: dict[str, Any]) -> None:
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "signals": signals,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    await _atomic_write_text(_structure_state_path(root_path), text)


def _should_refresh_structure(
    *,
    existing_state: dict[str, Any] | None,
    current_signals: dict[str, Any],
    structure_exists: bool,
    force: bool,
) -> list[str]:
    reasons: list[str] = []
    if force:
        reasons.append("force_refresh")
    if not structure_exists:
        reasons.append("structure_missing")

    previous_signals = existing_state.get("signals") if isinstance(existing_state, dict) else None
    if not isinstance(previous_signals, dict):
        reasons.append("state_missing")
        return reasons

    watched_keys = {
        "project_shape": "project_shape_changed",
        "top_level_paths": "top_level_paths_changed",
        "entrypoint_paths": "entrypoint_paths_changed",
        "area_prefixes": "area_prefixes_changed",
        "index_slugs": "index_slugs_changed",
        "fallback_sections": "fallback_sections_changed",
        "active_module_ids": "active_module_ids_changed",
        "active_module_slugs": "active_module_slugs_changed",
    }
    for key, reason in watched_keys.items():
        if previous_signals.get(key) != current_signals.get(key):
            reasons.append(reason)
    return reasons


def _summarize_registry_for_structure(
    registry: ModuleRegistry,
    *,
    max_items: int = 24,
) -> str:
    active_records = sorted(
        (record for record in registry.records.values() if record.status == "active"),
        key=lambda record: (record.slug, record.display_name),
    )
    lines = [f"- 活跃模块数：{len(active_records)}"]
    for record in active_records[:max_items]:
        file_preview = ", ".join(record.file_paths[:4]) or "(无覆盖文件)"
        if len(record.file_paths) > 4:
            file_preview += f", ... (+{len(record.file_paths) - 4} more)"
        lines.append(
            f"- {record.display_name} [{record.slug}] files={len(record.file_paths)} :: {file_preview}"
        )
    if len(active_records) > max_items:
        lines.append(f"- ... (其余 {len(active_records) - max_items} 个模块省略)")
    return "\n".join(lines)


def _build_structure_md_prompt(
    *,
    rule_structure_md: str,
    index_content: str,
    registry_summary: str,
) -> str:
    return "\n".join(
        [
            "你要基于规则骨架、模块导航和模块注册表，生成最终的 `structure.md`。",
            "目标是为主 Agent 提供全局结构导航，而不是复述所有细节。",
            "",
            "输出要求：",
            "- 输出 Markdown，不要代码块，不要额外解释。",
            "- 第一行必须是 `# 项目结构导航`。",
            "- 必须包含这些二级标题：`## 项目形态概览`、`## 顶层区域`、`## 关键入口候选`、`## 模块对齐提示`、`## 导航建议`。",
            "- 所有章节都优先使用短 bullet，不要写大段总括段落。",
            "- 表达必须保持克制，优先使用“候选 / 可优先查看 / 可视为 / 可能 / 线索 / 提示 / 建议”这类语气，避免过强断言。",
            "- 不要写“就是 / 明确是 / 一定是 / 清晰区分 / 负责全部 / 由...构成”这类结论化表述。",
            "- 以模块导航和注册表为主事实来源；规则骨架只作为弱提示，不要照抄其所有措辞。",
            "- 不要输出全量文件枚举，不要输出符号表，不要输出无根据的架构结论。",
            "",
            "风格示例：",
            "## 项目形态概览",
            "- 可初步视为多区域协作型项目；线索：顶层目录分布较散，且存在多个入口候选。",
            "",
            "## 顶层区域",
            "- `src/`：可优先作为主实现区域候选；线索：文件数较高，且与多个模块章节重叠。",
            "",
            "规则骨架：",
            rule_structure_md.strip(),
            "",
            "模块导航：",
            index_content.strip() or "(当前无 annotations/index.md)",
            "",
            "模块注册表摘要：",
            registry_summary.strip(),
        ]
    )


def _is_cautious_structure_markdown(content: str) -> bool:
    compacted = _compact_markdown_block(content)
    if not compacted.startswith("# 项目结构导航"):
        return False
    if not all(heading in compacted for heading in _STRUCTURE_REQUIRED_HEADINGS):
        return False
    # 至少出现若干弱提示标记，避免输出退化成强结论式总括。
    marker_hits = sum(compacted.count(marker) for marker in _STRUCTURE_HEDGE_MARKERS)
    if marker_hits < 6:
        return False
    current_heading = ""
    for line in compacted.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_heading = stripped
            continue
        if current_heading not in _STRUCTURE_CAUTIOUS_SECTION_HEADINGS:
            continue
        if not stripped.startswith("- "):
            continue
        if not any(marker in stripped for marker in _STRUCTURE_HEDGE_MARKERS):
            return False
    return True


async def _render_structure_markdown(
    entries: list[IndexEntry],
    *,
    root_path: Path,
    model: str | None,
    force_refresh: bool,
) -> tuple[str, dict[str, Any], list[str]]:
    structure_path = _structure_path(root_path)
    index_path = _annotation_index_path(root_path)
    index_content = ""
    if index_path.exists():
        try:
            index_content = await asyncio.to_thread(index_path.read_text, "utf-8")
        except Exception:
            index_content = ""
    header, index_sections = _split_index_md(index_content) if index_content.strip() else ("# 项目认知导航", [])
    _ = header  # 明示：这里只需要 sections

    registry = await ModuleRegistryManager(root_path).load()
    bundle = _build_structure_rule_bundle(entries, index_sections=index_sections)
    rule_structure_md = _render_rule_structure_md(bundle)
    current_signals = _build_structure_refresh_signals(
        bundle,
        index_sections=index_sections,
        registry=registry,
    )
    existing_state = await _load_structure_state(root_path)
    refresh_reasons = _should_refresh_structure(
        existing_state=existing_state,
        current_signals=current_signals,
        structure_exists=structure_path.exists(),
        force=force_refresh,
    )
    if not refresh_reasons and structure_path.exists():
        return (
            _compact_markdown_block(await asyncio.to_thread(structure_path.read_text, "utf-8")),
            current_signals,
            [],
        )

    content = rule_structure_md
    if index_content.strip():
        registry_summary = _summarize_registry_for_structure(registry)
        llm_content = await _llm_complete_text(
            _build_structure_md_prompt(
                rule_structure_md=rule_structure_md,
                index_content=_compact_markdown_block(index_content),
                registry_summary=registry_summary,
            ),
            system_prompt=(
                "你是代码库结构导航整理器。"
                "你的职责是将规则骨架与模块导航整合成一个谨慎、可读、可用于初始化提示词的全局结构页。"
            ),
            model=model,
            failure_log="structure.md 生成失败",
        )
        if llm_content:
            compacted = _compact_markdown_block(llm_content)
            if _is_cautious_structure_markdown(compacted):
                content = compacted + "\n"
            else:
                logger.warning("structure.md LLM 输出未通过谨慎性校验，回退到 rule structure")

    return content, current_signals, refresh_reasons


async def _write_structure_md(
    entries: list[IndexEntry],
    root_path: Path,
    *,
    model: str | None = None,
    force_refresh: bool = False,
) -> None:
    """在 annotation 生成后写入 structure.md。优先 LLM 归纳，失败时回退到规则骨架。"""
    content, current_signals, refresh_reasons = await _render_structure_markdown(
        entries,
        root_path=root_path,
        model=model,
        force_refresh=force_refresh,
    )
    if not refresh_reasons and _structure_path(root_path).exists():
        logger.info("structure.md 无需重算，沿用现有结果")
        return
    await _atomic_write_text(_structure_path(root_path), content)
    await _save_structure_state(root_path, current_signals)
    logger.info("structure.md 已更新: reasons=%s", ", ".join(refresh_reasons) or "unknown")


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


def _annotation_root_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/ 目录路径。"""
    return root_path / ".pce" / ANNOTATIONS_DIR


def _annotation_areas_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/areas/ 目录路径。"""
    return _annotation_root_dir(root_path) / ANNOTATIONS_AREAS_DIR


def _annotation_index_path(root_path: Path) -> Path:
    """返回 .pce/annotations/index.md 路径。"""
    return _annotation_root_dir(root_path) / ANNOTATIONS_INDEX_FILE


def _annotation_modules_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/modules/ 目录路径。"""
    return _annotation_root_dir(root_path) / ANNOTATIONS_MODULES_DIR


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


def _extract_text_windows(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if len(lines) <= _MISSING_FACT_FULL_LINES:
        return {"full": "\n".join(lines).strip()}

    head = "\n".join(lines[:_MISSING_FACT_WINDOW_LINES]).strip()
    mid_start = max(0, (len(lines) // 2) - (_MISSING_FACT_WINDOW_LINES // 2))
    middle = "\n".join(lines[mid_start: mid_start + _MISSING_FACT_WINDOW_LINES]).strip()
    tail = "\n".join(lines[-_MISSING_FACT_WINDOW_LINES:]).strip()
    return {
        "head": head,
        "middle": middle,
        "tail": tail,
    }


def _extract_markdown_hints(text: str) -> list[str]:
    hints: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            hints.append(stripped)
        if len(hints) >= 4:
            break
    return hints


def _extract_json_keys(text: str) -> list[str]:
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, dict):
        return [str(key) for key in list(payload.keys())[:8]]
    return []


def _extract_toml_keys(text: str) -> list[str]:
    try:
        payload = tomllib.loads(text)
    except Exception:
        return []
    if isinstance(payload, dict):
        return [str(key) for key in list(payload.keys())[:8]]
    return []


def _extract_yaml_like_keys(text: str) -> list[str]:
    keys: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:\s*", stripped)
        if match:
            keys.append(match.group(1))
        if len(keys) >= 8:
            break
    return keys


def _extract_file_type_hints(path: Path, text: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        headings = _extract_markdown_hints(text)
        return [f"Markdown headings: {', '.join(headings)}"] if headings else []
    if suffix == ".json":
        keys = _extract_json_keys(text)
        return [f"JSON top-level keys: {', '.join(keys)}"] if keys else []
    if suffix in {".toml"}:
        keys = _extract_toml_keys(text)
        return [f"TOML top-level keys: {', '.join(keys)}"] if keys else []
    if suffix in {".yml", ".yaml"}:
        keys = _extract_yaml_like_keys(text)
        return [f"YAML-like keys: {', '.join(keys)}"] if keys else []
    return []


async def _build_missing_entry_fact(entry: IndexEntry, root_path: Path) -> dict[str, Any]:
    rel_path = str(entry.file_meta.path)
    abs_path = root_path / rel_path
    fact: dict[str, Any] = {
        "path": rel_path,
        "language": entry.file_meta.language,
        "loc": entry.file_meta.loc,
        "symbol_summary": [
            f"{sym.name}({sym.kind.value})" for sym in entry.symbols[:10]
        ] or ["无显式符号"],
    }

    try:
        text = await asyncio.to_thread(abs_path.read_text, "utf-8")
    except Exception:
        fact["content_windows"] = {"unavailable": "(文件内容读取失败或不可解码)"}
        return fact

    type_hints = _extract_file_type_hints(abs_path, text)
    if type_hints:
        fact["type_hints"] = type_hints
    fact["content_windows"] = _extract_text_windows(text)
    return fact


def _format_missing_entry_fact_block(fact: dict[str, Any]) -> str:
    lines = [
        f"- path: {fact['path']}",
        f"  language: {fact['language']}",
        f"  loc: {fact['loc']}",
        f"  symbol_summary: {', '.join(fact['symbol_summary'])}",
    ]
    type_hints = fact.get("type_hints") or []
    for hint in type_hints:
        lines.append(f"  type_hint: {hint}")

    windows = fact.get("content_windows") or {}
    if "full" in windows:
        lines.append("  snippet_full:")
        lines.append("```text")
        lines.append(windows["full"])
        lines.append("```")
    else:
        for key in ("head", "middle", "tail", "unavailable"):
            if key not in windows:
                continue
            lines.append(f"  snippet_{key}:")
            lines.append("```text")
            lines.append(windows[key])
            lines.append("```")
    return "\n".join(lines)


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


def _build_missing_coverage_repair_prompt(
    entry_facts: list[dict[str, Any]],
    index_content: str,
    known_slugs: set[str],
) -> str:
    fact_blocks = [_format_missing_entry_fact_block(fact) for fact in entry_facts]
    return "\n".join(
        [
            "当前项目认知导航已有第一轮模块划分，但仍有少量文件未被任何模块覆盖。",
            "请只处理这些遗漏文件，优先判断它们可能是什么性质，再决定是否应并入已有模块；只有确有必要时才创建新的补充模块。",
            "注意：这里的 kind 与 action 都只是导航性推论，不是事实断言；若证据不足，优先 fallback。",
            "",
            "可用动作：",
            "- attach: 并入已有模块（必须使用现有 module_slug）",
            "- create: 创建一个新的补充模块（可让多个文件共享同一个 module_name）",
            "- fallback: 暂不可靠归属，留给最终 fallback",
            "",
            "kind 可选值：",
            "- implementation / config / documentation / test / resource / shell / entrypoint / unknown",
            "",
            "现有导航：",
            index_content.strip(),
            "",
            "现有 module_slug 列表：",
            ", ".join(sorted(known_slugs)) or "(无)",
            "",
            "遗漏文件 facts 包：",
            *fact_blocks,
            "",
            "输出要求：",
            '- 只输出 JSON，例如 {"decisions":[{"path":"src/main.py","kind":"entrypoint","action":"attach","module_slug":"agent-core"}]}',
            "- path 必须来自 facts 包中的原文路径",
            "- kind 必填，且必须来自给定 kind 列表",
            "- attach 只能填现有 module_slug",
            "- create 时必须提供 module_name，可选 responsibility；若多个文件属于同一新模块，请重复使用同一个 module_name",
            "- 如果不确定，就用 fallback，不要勉强归类",
            "- 不要输出 Markdown、代码块或解释",
        ]
    )


def _build_additional_section(
    module_name: str,
    file_paths: list[str],
    *,
    responsibility: str | None = None,
    slug: str | None = None,
) -> dict[str, Any]:
    normalized = _dedupe_keep_order(file_paths)
    effective_slug = slug or ModuleRegistryManager.normalize_slug(module_name, file_paths=normalized)
    responsibility_line = (
        responsibility.strip()
        if responsibility and responsibility.strip()
        else "职责：该章节由二阶段补归属逻辑创建，用于承接第一轮主模块划分后仍遗漏、但具备独立语义的文件集合。"
    )
    return {
        "name": module_name,
        "slug": effective_slug,
        "file_paths": normalized,
        "body_lines": [
            f"文件：{', '.join(normalized)}",
            responsibility_line if responsibility_line.startswith("职责：") else f"职责：{responsibility_line}",
            f"详细认知：.pce/annotations/modules/{effective_slug}.md",
        ],
    }


def _ensure_unique_section_slug(base_slug: str, existing_slugs: set[str]) -> str:
    if base_slug not in existing_slugs:
        return base_slug
    suffix = 2
    while f"{base_slug}-{suffix}" in existing_slugs:
        suffix += 1
    return f"{base_slug}-{suffix}"


def _parse_missing_coverage_repair_decisions(
    content: str,
    *,
    candidate_paths: set[str],
    known_slugs: set[str],
) -> list[dict[str, str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    raw_items = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        return []

    decisions: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        if not path or path not in candidate_paths or path in seen_paths:
            continue
        if action not in {"attach", "create", "fallback"}:
            continue
        record: dict[str, str] = {"path": path, "action": action}
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in _MISSING_KIND_VALUES:
            continue
        record["kind"] = kind
        if action == "attach":
            slug = str(item.get("module_slug") or "").strip()
            if slug not in known_slugs:
                continue
            record["module_slug"] = slug
        elif action == "create":
            module_name = str(item.get("module_name") or "").strip()
            if not module_name:
                continue
            record["module_name"] = module_name
            responsibility = str(item.get("responsibility") or "").strip()
            if responsibility:
                record["responsibility"] = responsibility
        seen_paths.add(path)
        decisions.append(record)
    return decisions


def _apply_missing_coverage_repair_decisions(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
    decisions: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    augmented = list(sections)
    sections_by_slug = {section["slug"]: section for section in augmented}
    changed_slugs: set[str] = set()

    assigned_before = {
        file_path
        for section in augmented
        for file_path in _dedupe_keep_order(section.get("file_paths", []))
        if file_path in entries_map
    }
    unassigned_paths = sorted(path for path in entries_map if path not in assigned_before)
    create_groups: dict[str, dict[str, Any]] = {}

    for decision in decisions:
        path = decision["path"]
        action = decision["action"]
        if path not in unassigned_paths:
            continue
        if action == "attach":
            slug = decision["module_slug"]
            section = sections_by_slug.get(slug)
            if section is None:
                continue
            _update_section_file_list(section, section.get("file_paths", []) + [path])
            changed_slugs.add(slug)
            continue
        if action == "create":
            module_name = decision["module_name"]
            responsibility = decision.get("responsibility") or ""
            kind = decision.get("kind") or "unknown"
            group = create_groups.setdefault(
                module_name,
                {"paths": [], "responsibility": responsibility, "kind": kind},
            )
            group["paths"].append(path)
            if not group["responsibility"] and responsibility:
                group["responsibility"] = responsibility
            if not group["kind"] and kind:
                group["kind"] = kind

    existing_slugs = set(sections_by_slug.keys())
    for module_name, payload in create_groups.items():
        paths = _dedupe_keep_order(payload["paths"])
        if not paths:
            continue
        slug = _ensure_unique_section_slug(
            ModuleRegistryManager.normalize_slug(module_name, file_paths=paths),
            existing_slugs,
        )
        section = _build_additional_section(
            module_name,
            paths,
            responsibility=payload["responsibility"]
            or f"该章节由二阶段补归属逻辑创建，当前更像 `{payload['kind']}` 类型的文件集合，仅供导航参考。",
            slug=slug,
        )
        augmented.append(section)
        sections_by_slug[slug] = section
        existing_slugs.add(slug)
        changed_slugs.add(slug)

    final_assigned = {
        file_path
        for section in augmented
        for file_path in _dedupe_keep_order(section.get("file_paths", []))
        if file_path in entries_map
    }
    remaining = sorted(path for path in entries_map if path not in final_assigned)
    return augmented, changed_slugs, remaining


async def _supplement_unassigned_entries_with_llm(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
    *,
    root_path: Path,
    model: str | None,
) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    def _collect_unassigned_paths(current_sections: list[dict[str, Any]]) -> list[str]:
        assigned = {
            file_path
            for section in current_sections
            for file_path in _dedupe_keep_order(section.get("file_paths", []))
            if file_path in entries_map
        }
        return sorted(path for path in entries_map if path not in assigned)

    current_sections = list(sections)
    pending = _collect_unassigned_paths(current_sections)
    if not pending or model is None:
        return current_sections, set(), pending

    all_changed_slugs: set[str] = set()

    while pending:
        known_slugs = {section["slug"] for section in current_sections}
        if not known_slugs:
            break

        batch_paths = pending[:_MISSING_COVERAGE_REPAIR_BATCH_SIZE]
        batch_entries = [entries_map[path] for path in batch_paths if path in entries_map]
        batch_facts = await asyncio.gather(
            *[_build_missing_entry_fact(entry, root_path) for entry in batch_entries]
        )
        index_content = _render_index_md("# 项目认知导航", current_sections)
        content = await _llm_complete_text(
            _build_missing_coverage_repair_prompt(batch_facts, index_content, known_slugs),
            system_prompt=(
                "你是代码库模块补归属助手。"
                "你的职责是在第一轮模块划分之后，只处理遗漏文件，先判断其性质(kind)，再谨慎决定 attach/create/fallback。"
            ),
            model=model,
            failure_log="遗漏文件二阶段补归属失败",
        )
        if not content:
            break

        decisions = _parse_missing_coverage_repair_decisions(
            content,
            candidate_paths=set(batch_paths),
            known_slugs=known_slugs,
        )
        if not decisions:
            pending = pending[_MISSING_COVERAGE_REPAIR_BATCH_SIZE:]
            continue

        current_sections, changed_slugs, remaining = _apply_missing_coverage_repair_decisions(
            current_sections,
            entries_map,
            decisions,
        )
        all_changed_slugs.update(changed_slugs)
        pending = remaining

        # 若本批没有实际消化任何文件，则避免死循环，继续处理下一批。
        if not changed_slugs:
            pending = pending[_MISSING_COVERAGE_REPAIR_BATCH_SIZE:]

    return current_sections, all_changed_slugs, _collect_unassigned_paths(current_sections)


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
    def _normalize_body_lines(lines: list[str]) -> list[str]:
        normalized: list[str] = []
        previous_blank = False
        for raw in lines:
            line = raw.rstrip()
            blank = not line.strip()
            if blank:
                if previous_blank or not normalized:
                    continue
                previous_blank = True
                normalized.append("")
                continue
            previous_blank = False
            normalized.append(line)
        while normalized and not normalized[-1].strip():
            normalized.pop()
        return normalized

    lines = [header.strip() or "# 项目认知导航"]
    for section in sections:
        lines.extend(["", f"## {section['name']}"])
        lines.extend(_normalize_body_lines(section["body_lines"]))
    return "\n".join(lines).rstrip() + "\n"


async def _cleanup_stale_module_docs(modules_dir: Path, section_slugs: set[str]) -> None:
    """删除未被当前 index sections 引用的模块文档。"""
    if not modules_dir.exists():
        return
    expected = {f"{slug}.md" for slug in section_slugs}
    for stale in modules_dir.glob("*.md"):
        if stale.name not in expected:
            await asyncio.to_thread(stale.unlink, missing_ok=True)


async def _cleanup_stale_annotation_docs(
    root_path: Path,
    expected_files: set[Path],
) -> None:
    """根据期望文件集合清理 annotations 下的过期文档。

    兼容新布局（areas/*.md + navigation_tree.json）和旧布局残留。
    expected_files 中的路径均为相对于 .pce/annotations/ 的路径。
    """
    annotation_root = _annotation_root_dir(root_path)
    if not annotation_root.exists():
        return

    expected_posix = {p.as_posix() for p in expected_files}

    # 收集所有候选文件（含新布局和旧布局残留）
    candidates: list[Path] = []
    for pattern in (
        "index.md",
        "navigation_tree.json",
        "modules/*.md",
        "areas/*.md",
        "areas/*/README.md",
        "areas/*/modules/*.md",
    ):
        candidates.extend(annotation_root.glob(pattern))

    for stale in candidates:
        rel = stale.relative_to(annotation_root).as_posix()
        if rel not in expected_posix:
            await asyncio.to_thread(stale.unlink, missing_ok=True)

    # 清理空的 areas 子目录（仅限 areas/ 下，不触碰其它目录）
    areas_dir = _annotation_areas_dir(root_path)
    if areas_dir.exists():
        for area_subdir in sorted(areas_dir.iterdir(), reverse=True):
            if not area_subdir.is_dir():
                continue
            # 先清理空的 modules/ 子目录
            modules_subdir = area_subdir / "modules"
            if modules_subdir.is_dir() and not any(modules_subdir.iterdir()):
                try:
                    await asyncio.to_thread(modules_subdir.rmdir)
                except OSError:
                    pass
            # 再清理空的 area 目录
            if not any(area_subdir.iterdir()):
                try:
                    await asyncio.to_thread(area_subdir.rmdir)
                except OSError:
                    pass


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
    """从已有模块文档中恢复章节信息，兼容 flat/tree 两种布局。

    用途：
    - 当 LLM 生成的 index.md 漏掉部分现有模块时，避免直接退化为粗粒度 fallback 模块
    - 将已经存在的细粒度认知文档重新纳入 section 体系，后续再由 registry 稳定化

    优先级：tree 布局（areas/*/modules/*.md）> flat 布局（modules/*.md），
    同 stem 的文档一旦在 tree 中命中就不再回退读取 flat 版本。
    """
    seen_stems: set[str] = set()
    seen_slugs: set[str] = set()
    sections: list[dict[str, Any]] = []

    # 分组扫描，显式保证 tree-first
    scan_groups: list[list[Path]] = []

    # tree 布局优先：areas/*/modules/*.md
    areas_dir = _annotation_areas_dir(root_path)
    if areas_dir.exists():
        scan_groups.append(sorted(areas_dir.glob("*/modules/*.md")))

    # flat 布局兜底：modules/*.md
    modules_dir = _annotation_modules_dir(root_path)
    if modules_dir.exists():
        scan_groups.append(sorted(modules_dir.glob("*.md")))

    for group in scan_groups:
        for module_path in group:
            # stem 早期去重：tree 命中后跳过 flat 同名文档
            stem = module_path.stem
            if stem in seen_stems:
                continue
            try:
                raw = module_path.read_text("utf-8")
            except Exception:
                logger.warning(f"读取已有模块文档失败，跳过恢复: {module_path}")
                continue
            section = _parse_existing_module_doc_to_section(raw, stem)
            if section is not None and section["slug"] not in seen_slugs:
                seen_stems.add(stem)
                seen_slugs.add(section["slug"])
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


def _normalize_section_name(name: str) -> str:
    normalized = name.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def _section_body_signal(section: dict[str, Any]) -> int:
    score = 0
    for line in section.get("body_lines", []):
        stripped = line.strip()
        if stripped.startswith("职责："):
            score += 3
        elif stripped.startswith("文件：") or stripped.startswith("详细认知："):
            score += 1
        elif stripped:
            score += 1
    return score


def _section_overlap_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_files = set(left.get("file_paths", []))
    right_files = set(right.get("file_paths", []))
    if not left_files or not right_files:
        return 0.0
    intersection = len(left_files & right_files)
    union = len(left_files | right_files)
    return intersection / union if union else 0.0


def _should_merge_sections(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = _normalize_section_name(left["name"])
    right_name = _normalize_section_name(right["name"])
    left_files = set(left.get("file_paths", []))
    right_files = set(right.get("file_paths", []))
    if left_name == right_name:
        return True

    overlap = _section_overlap_score(left, right)
    left_tokens = set(left_name.split())
    right_tokens = set(right_name.split())
    common = left_tokens & right_tokens
    if overlap >= 0.45:
        return bool(common) and (left_name in right_name or right_name in left_name or len(common) >= 2)

    subset_relation = (
        (left_files and left_files <= right_files)
        or (right_files and right_files <= left_files)
    )
    return subset_relation and bool(common) and (left_name in right_name or right_name in left_name)


def _prefer_section(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rank = (
        1 if not _is_fallback_section(left) else 0,
        _section_body_signal(left),
        len(left.get("file_paths", [])),
    )
    right_rank = (
        1 if not _is_fallback_section(right) else 0,
        _section_body_signal(right),
        len(right.get("file_paths", [])),
    )
    return left if left_rank >= right_rank else right


def _dedupe_semantic_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    for section in sections:
        merged = False
        for idx, existing in enumerate(deduped):
            if not _should_merge_sections(existing, section):
                continue
            preferred = _prefer_section(existing, section)
            other = section if preferred is existing else existing
            _update_section_file_list(
                preferred,
                preferred.get("file_paths", []) + other.get("file_paths", []),
            )
            deduped[idx] = preferred
            merged = True
            break
        if not merged:
            deduped.append(section)
    return deduped


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


# ============================================================================
# 树状导航生成（facts 构建 / 生成 / 校验 / 修复 / fallback / 渲染）
# ============================================================================


import hashlib

from .models import AreaRecord, NavigationTree

NAVIGATION_TREE_FILE = "navigation_tree.json"
NAVIGATION_FALLBACK_AREA_SLUG = "fallback"
NAVIGATION_FALLBACK_AREA_NAME = "未分类（Fallback）"


def _extract_section_summary(section: dict[str, Any]) -> str:
    """从扁平 section 中提取一段适合导航树展示的摘要。"""
    for line in section.get("body_lines", []):
        match = re.match(r"^职责[:：]\s*(.+)$", line.strip())
        if match and match.group(1).strip():
            return match.group(1).strip()
    for line in section.get("body_lines", []):
        stripped = line.strip()
        if not stripped or re.match(r"^(文件|详细认知)[:：]\s*", stripped):
            continue
        return stripped
    return ""


def _compute_source_digest(sections: list[dict[str, Any]]) -> str:
    """基于稳定 slug 与文件签名计算导航源指纹（纯 Python 侧计算）。"""
    payload = [
        {
            "slug": section["slug"],
            "files": sorted(_dedupe_keep_order(section.get("file_paths", []))),
        }
        for section in sorted(sections, key=lambda s: s["slug"])
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_navigation_facts(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
    root_path: Path,
) -> dict[str, Any]:
    """构建三层导航生成所需的受控事实包（有固定预算）。"""
    # 项目级 facts
    top_dir_entries: dict[str, list[IndexEntry]] = defaultdict(list)
    language_counter: Counter[str] = Counter()
    for entry in entries_map.values():
        parts = Path(entry.file_meta.path).parts
        top_dir = f"{parts[0]}/" if len(parts) > 1 else "./"
        top_dir_entries[top_dir].append(entry)
        language_counter[entry.file_meta.language] += 1

    # 顶层目录摘要（限宽度 8）
    top_level_dirs = [
        {
            "path": directory,
            "file_count": len(items),
            "languages": _format_language_summary(items, max_items=3),
            "representatives": _select_representative_paths(items, max_items=3),
        }
        for directory, items in sorted(
            top_dir_entries.items(), key=lambda x: (-len(x[1]), x[0])
        )[:8]
    ]

    # 模块级 facts（预算受控）
    module_facts: list[dict[str, Any]] = []
    for section in sorted(sections, key=lambda s: s["slug"]):
        file_paths = [
            p for p in _dedupe_keep_order(section.get("file_paths", []))
            if p in entries_map
        ]
        module_entries = [entries_map[p] for p in file_paths]
        summary = _extract_section_summary(section)
        if len(summary) > 150:
            summary = summary[:147].rstrip() + "..."
        module_facts.append({
            "slug": section["slug"],
            "display_name": section["name"],
            "representative_files": file_paths[:5],
            "key_symbols": _extract_module_key_symbols(module_entries)[:6],
            "summary": summary,
        })

    primary_language = (
        language_counter.most_common(1)[0][0] if language_counter else "unknown"
    )
    # 区域候选 hints（弱提示，可置空不影响主流程）
    area_hints = _select_core_areas(top_dir_entries)

    return {
        "project": {
            "name": root_path.name,
            "file_count": len(entries_map),
            "primary_language": primary_language,
            "top_level_dirs": top_level_dirs,
        },
        "modules": module_facts,
        "area_hints": area_hints,
        "source_digest": _compute_source_digest(sections),
    }


def _validate_navigation_tree(
    tree: NavigationTree, active_slugs: set[str]
) -> list[str]:
    """校验导航树结构完整性，返回错误列表（空列表表示通过）。"""
    errors: list[str] = []
    area_slug_seen: set[str] = set()
    mounted: dict[str, str] = {}
    fallback_areas = [a for a in tree.areas if a.is_fallback]

    if not 2 <= len(tree.areas) <= 8:
        errors.append(f"area 数量必须在 2-8 之间，当前为 {len(tree.areas)}")

    for area in tree.areas:
        if area.slug in area_slug_seen:
            errors.append(f"区域 slug 重复: {area.slug}")
        area_slug_seen.add(area.slug)

        if not area.is_fallback and not area.module_slugs:
            errors.append(f"非 fallback 区域不能为空: {area.slug}")

        for ms in area.module_slugs:
            if ms not in active_slugs:
                errors.append(f"区域 {area.slug} 引用了不存在模块: {ms}")
                continue
            prev = mounted.get(ms)
            if prev is not None and prev != area.slug:
                errors.append(
                    f"模块重复挂载: {ms} 同时出现在 {prev} / {area.slug}"
                )
                continue
            mounted[ms] = area.slug

    if len(fallback_areas) != 1:
        errors.append(
            f"fallback 区域必须恰好 1 个，当前为 {len(fallback_areas)}"
        )
    elif fallback_areas[0].slug != tree.fallback_area_slug:
        errors.append(
            f"fallback_area_slug ({tree.fallback_area_slug}) "
            f"与 fallback 记录 ({fallback_areas[0].slug}) 不一致"
        )

    missing = sorted(active_slugs - set(mounted))
    if missing:
        errors.append("存在未覆盖模块: " + ", ".join(missing[:12]))

    return errors


def _repair_navigation_tree(
    tree: NavigationTree, active_slugs: set[str]
) -> NavigationTree:
    """自动修复导航树中的常见结构问题。"""
    active = set(active_slugs)
    used_modules: set[str] = set()
    used_area_slugs: set[str] = set()
    repaired_areas: list[AreaRecord] = []
    fallback_modules: list[str] = []
    fallback_order: list[str] = []
    fallback_prefixes: list[str] = []
    fallback_summary = ""

    def _uniq_slug(base: str) -> str:
        slug = base or "area"
        if slug not in used_area_slugs:
            used_area_slugs.add(slug)
            return slug
        idx = 2
        while f"{slug}-{idx}" in used_area_slugs:
            idx += 1
        uniq = f"{slug}-{idx}"
        used_area_slugs.add(uniq)
        return uniq

    for area in tree.areas:
        mods = [
            s for s in _dedupe_keep_order(area.module_slugs)
            if s in active and s not in used_modules
        ]
        for s in mods:
            used_modules.add(s)

        order = [
            s for s in _dedupe_keep_order(area.recommended_order)
            if s in set(mods)
        ]
        for s in mods:
            if s not in order:
                order.append(s)

        if area.is_fallback:
            fallback_modules.extend(mods)
            fallback_order.extend(order)
            fallback_prefixes.extend(area.source_prefixes)
            fallback_summary = fallback_summary or area.summary
            continue

        if not mods:
            continue

        repaired_areas.append(area.model_copy(update={
            "slug": _uniq_slug(area.slug),
            "module_slugs": mods,
            "recommended_order": order,
        }))

    # 缺失模块塞入 fallback
    missing = [s for s in sorted(active) if s not in used_modules]
    fallback_modules = _dedupe_keep_order([*fallback_modules, *missing])
    fallback_order = _dedupe_keep_order([*fallback_order, *fallback_modules])

    # area 数量上限 7（+1 fallback = 8）
    if len(repaired_areas) > 7:
        for overflow in repaired_areas[7:]:
            fallback_modules.extend(overflow.module_slugs)
            fallback_order.extend(overflow.recommended_order)
        repaired_areas = repaired_areas[:7]
        fallback_modules = _dedupe_keep_order(fallback_modules)
        fallback_order = _dedupe_keep_order(
            [*fallback_order, *fallback_modules]
        )

    fb_slug = tree.fallback_area_slug or NAVIGATION_FALLBACK_AREA_SLUG
    if fb_slug in used_area_slugs:
        fb_slug = _uniq_slug(NAVIGATION_FALLBACK_AREA_SLUG)
    else:
        used_area_slugs.add(fb_slug)

    fb_area = AreaRecord(
        slug=fb_slug,
        display_name=NAVIGATION_FALLBACK_AREA_NAME,
        summary=fallback_summary or "承接暂时无法稳定归入主区域的模块。",
        module_slugs=fallback_modules,
        recommended_order=fallback_order,
        source_prefixes=_dedupe_keep_order(fallback_prefixes),
        is_fallback=True,
    )

    proj_summary = (tree.project_summary or "").strip()
    if not proj_summary:
        proj_summary = (
            f"项目当前共 {len(active)} 个模块，已整理为区域化导航入口。"
        )

    return tree.model_copy(update={
        "generated_at": datetime.now(UTC),
        "project_summary": proj_summary,
        "fallback_area_slug": fb_area.slug,
        "areas": [*repaired_areas, fb_area],
    })


def _build_fallback_navigation_tree(
    sections: list[dict[str, Any]],
    active_slugs: set[str],
) -> NavigationTree:
    """按目录前缀分组构建确定性 fallback 导航树。"""
    active = set(active_slugs)
    groups: dict[str, list[str]] = defaultdict(list)

    for section in sections:
        slug = section["slug"]
        if slug not in active:
            continue
        file_paths = _dedupe_keep_order(section.get("file_paths", []))
        if not file_paths:
            groups["./"].append(slug)
            continue
        head = (
            Path(file_paths[0]).parts[0]
            if len(Path(file_paths[0]).parts) > 1
            else "./"
        )
        groups[head].append(slug)

    ranked = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))

    areas: list[AreaRecord] = []
    used: set[str] = set()
    used_area_slugs: set[str] = set()

    def _uniq_area_slug(base: str) -> str:
        slug = base or "area"
        if slug not in used_area_slugs:
            used_area_slugs.add(slug)
            return slug
        idx = 2
        while f"{slug}-{idx}" in used_area_slugs:
            idx += 1
        uniq = f"{slug}-{idx}"
        used_area_slugs.add(uniq)
        return uniq

    for prefix, slugs in ranked[:7]:
        ordered = _dedupe_keep_order(slugs)
        used.update(ordered)
        name = (
            "项目根目录"
            if prefix == "./"
            else prefix.rstrip("/").replace("-", " ").replace("_", " ").title()
        )
        raw_slug = name.lower().replace(" ", "-").replace("/", "-").strip("-")
        area_slug = _uniq_area_slug(raw_slug)
        areas.append(AreaRecord(
            slug=area_slug,
            display_name=name,
            summary=f"主要覆盖 `{prefix}` 前缀下的模块。",
            module_slugs=ordered,
            recommended_order=ordered,
            source_prefixes=[prefix],
            is_fallback=False,
        ))

    missing = [s for s in sorted(active) if s not in used]
    fb_area = AreaRecord(
        slug=NAVIGATION_FALLBACK_AREA_SLUG,
        display_name=NAVIGATION_FALLBACK_AREA_NAME,
        summary="承接零散目录或当前无法稳定分组的模块。",
        module_slugs=missing,
        recommended_order=missing,
        source_prefixes=[],
        is_fallback=True,
    )

    # 至少需要 1 个非 fallback area
    if not areas:
        seed = sorted(active)[:min(len(active), 8)]
        areas.append(AreaRecord(
            slug="core",
            display_name="核心模块",
            summary="目录分群信号不足时的确定性主区域。",
            module_slugs=seed,
            recommended_order=seed,
            source_prefixes=[],
            is_fallback=False,
        ))
        fb_area = fb_area.model_copy(update={
            "module_slugs": [s for s in sorted(active) if s not in set(seed)],
            "recommended_order": [
                s for s in sorted(active) if s not in set(seed)
            ],
        })

    return NavigationTree(
        generated_at=datetime.now(UTC),
        project_summary=(
            f"项目当前共 {len(active)} 个模块，"
            "已按目录前缀生成确定性区域入口。"
        ),
        fallback_area_slug=fb_area.slug,
        source_digest=_compute_source_digest(sections),
        areas=[*areas, fb_area],
    )


async def _generate_navigation_tree(
    facts: dict[str, Any],
    active_slugs: set[str],
    model: str | None,
) -> NavigationTree | None:
    """一次 LLM 调用生成导航树 JSON；失败时给一次修订机会。"""
    system_prompt = (
        "你是 PCE 认知导航规划器。基于给定 facts 输出 JSON，不得输出解释文字。\n"
        "区域是介于'项目整体'和'具体模块'之间的中层认知单元。\n"
        "区域划分优先依据职责边界，其次才参考目录邻近性。\n"
        "区域名称应反映功能职责，不要用目录路径做名称。\n"
        "fallback 区域仅收纳无法可靠分类的模块，不应成为最大区域。"
    )

    attempt_prompt = "\n".join([
        "请基于以下受控 facts 生成三层导航树 JSON。",
        "输出字段: version, project_summary, fallback_area_slug, source_digest, areas。",
        "areas[*] 字段: slug, display_name, summary, module_slugs, "
        "recommended_order, source_prefixes, is_fallback。",
        "生成 2-8 个区域，其中恰好 1 个 is_fallback=true。",
        "每个模块恰好出现在一个区域中，完整覆盖所有模块。",
        "",
        json.dumps(facts, ensure_ascii=False, indent=2),
    ])

    content = await _llm_complete_text(
        attempt_prompt,
        system_prompt=system_prompt,
        model=model,
        failure_log="导航树生成失败",
    )
    if not content:
        return None

    def _try_parse(raw: str) -> tuple[NavigationTree | None, list[str]]:
        """尝试解析 LLM 输出为 NavigationTree，返回 (tree, errors)。"""
        try:
            payload = json.loads(raw)
            payload["source_digest"] = str(facts["source_digest"])
            payload["generated_at"] = datetime.now(UTC).isoformat()
            tree = NavigationTree.model_validate(payload)
            tree = _repair_navigation_tree(tree, active_slugs)
            errs = _validate_navigation_tree(tree, active_slugs)
            return (tree if not errs else None), errs
        except Exception as e:
            return None, [f"JSON / schema 校验失败: {e}"]

    # 第一次尝试解析
    tree, feedback = _try_parse(content)
    if tree is not None:
        return tree

    # 给一次修订机会
    content = await _llm_complete_text(
        "\n".join([
            "请根据以下校验错误修订导航树 JSON，只输出修订后的完整 JSON：",
            *feedback,
            "",
            json.dumps(facts, ensure_ascii=False, indent=2),
        ]),
        system_prompt=system_prompt,
        model=model,
        failure_log="导航树修订失败",
    )
    if not content:
        return None

    # 解析修订结果
    tree, _ = _try_parse(content)
    return tree


def _render_hierarchical_index_md(
    tree: NavigationTree,
    sections_by_slug: dict[str, dict[str, Any]],
) -> str:
    """渲染项目级 index.md — 只承载项目概览与区域入口，不展开模块细节。"""
    total_modules = sum(len(a.module_slugs) for a in tree.areas)
    lines = [
        "# 项目认知导航",
        "",
        "## 项目概览",
        tree.project_summary or "当前暂无项目级摘要。",
        "",
        f"- 区域数：{len(tree.areas)}",
        f"- 模块数：{total_modules}",
        "",
        "## 区域入口",
    ]
    for area in tree.areas:
        previews = [
            sections_by_slug.get(s, {}).get("name", s)
            for s in area.module_slugs[:3]
        ]
        preview_text = (
            f"；示例模块：{'、'.join(previews)}" if previews else ""
        )
        summary = area.summary or "暂无区域摘要。"
        lines.append(
            f"- [{area.display_name}](areas/{area.slug}.md)：{summary}"
            f"（{len(area.module_slugs)} 个模块{preview_text}）"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_area_md(
    area: AreaRecord,
    sections_by_slug: dict[str, dict[str, Any]],
) -> str:
    """渲染区域文档 areas/{slug}.md — 只引用模块级文档，不复述模块正文。"""
    lines = [
        f"# {area.display_name}",
        "",
        "## 区域说明",
        area.summary or "当前暂无区域说明。",
        "",
        "## 模块列表",
    ]
    if area.module_slugs:
        for slug in area.module_slugs:
            section = sections_by_slug.get(slug, {})
            name = section.get("name", slug)
            summary = _extract_section_summary(section) or "暂无模块摘要。"
            lines.append(
                f"- [{name}](../modules/{slug}.md) (`{slug}`)：{summary}"
            )
    else:
        lines.append("- 当前无挂载模块。")

    lines.extend(["", "## 推荐阅读顺序"])
    ordered = area.recommended_order or area.module_slugs
    if ordered:
        for idx, slug in enumerate(ordered, start=1):
            name = sections_by_slug.get(slug, {}).get("name", slug)
            lines.append(f"{idx}. [{name}](../modules/{slug}.md)")
    else:
        lines.append("1. 当前无推荐顺序。")

    return "\n".join(lines).rstrip() + "\n"


async def _cleanup_stale_area_docs(
    areas_dir: Path, valid_slugs: set[str]
) -> None:
    """清理不再被引用的区域文档。"""
    if not areas_dir.exists():
        return
    expected = {f"{slug}.md" for slug in valid_slugs}
    for child in areas_dir.iterdir():
        if child.is_file() and child.suffix == ".md" and child.name not in expected:
            await asyncio.to_thread(child.unlink, missing_ok=True)


def _navigation_tree_path(root_path: Path) -> Path:
    """返回 .pce/annotations/navigation_tree.json 路径。"""
    return _annotation_root_dir(root_path) / NAVIGATION_TREE_FILE


def _annotation_areas_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/areas/ 目录路径。"""
    return _annotation_root_dir(root_path) / ANNOTATIONS_AREAS_DIR


async def _persist_navigation_area_cache(
    root_path: Path, tree: NavigationTree
) -> None:
    """将 area_slug 写回 module registry 的缓存字段。"""
    manager = ModuleRegistryManager(root_path)
    registry = await manager.load()
    module_area_map = {
        ms: area.slug
        for area in tree.areas
        for ms in area.module_slugs
    }
    changed = False
    for record in registry.records.values():
        if record.status != "active":
            continue
        next_area = module_area_map.get(record.slug)
        if record.area_slug != next_area:
            record.area_slug = next_area
            changed = True
    if changed:
        await manager.save(registry)


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
    """并发生成并写入模块认知文档，失败时使用回退内容。"""
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
    """全量生成并写入三层树状认知导航。"""
    index_path = _annotation_index_path(root_path)
    areas_dir = _annotation_areas_dir(root_path)
    modules_dir = _annotation_modules_dir(root_path)

    if not entries:
        await _atomic_write_text(
            index_path, "# 项目认知导航\n\n当前暂无可导航模块。\n"
        )
        await _cleanup_stale_area_docs(areas_dir, set())
        # 删除残留的 navigation_tree.json
        tree_path = _navigation_tree_path(root_path)
        if tree_path.exists():
            await asyncio.to_thread(tree_path.unlink, missing_ok=True)
        await _cleanup_stale_annotation_docs(
            root_path, {Path(ANNOTATIONS_INDEX_FILE)}
        )
        return

    # ── 阶段 1：模块发现与稳定化（保留原有逻辑不动）──
    index_content = await _generate_index_md(entries, project_meta, model=model)
    if not index_content:
        logger.warning("认知导航 index.md 不可用，使用索引级回退模板")
        index_content = _build_fallback_index_md(entries)

    entries_map = {str(e.file_meta.path): e for e in entries}
    _, sections = _split_index_md(index_content)
    sections = _dedupe_sections_by_slug(sections)
    sections = _merge_missing_sections(
        sections,
        await _load_sections_from_existing_module_docs(root_path),
    )
    sections, _, _ = await _supplement_unassigned_entries_with_llm(
        sections,
        entries_map,
        root_path=root_path,
        model=model,
    )
    sections = _dedupe_semantic_sections(sections)
    sections = _append_fallback_sections_for_unassigned_entries(
        sections, entries_map
    )
    sections = await _stabilize_sections_with_registry(
        sections,
        root_path=root_path,
        entries_map=entries_map,
    )
    valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    if not valid_sections:
        logger.warning("认知导航解析为空，回退到结构化模板")
        index_content = _build_fallback_index_md(entries)
        _, sections = _split_index_md(index_content)
        valid_sections, module_specs = _prepare_module_specs(
            sections, entries_map
        )

    # ── 阶段 2：树状导航生成 ──
    sections_by_slug = {s["slug"]: s for s in valid_sections}
    active_slugs = {slug for _, slug, _ in module_specs}
    facts = _build_navigation_facts(valid_sections, entries_map, root_path)

    tree = await _generate_navigation_tree(facts, active_slugs, model=model)
    if tree is None:
        logger.warning("导航树生成失败，回退到确定性 fallback 方案")
        tree = _build_fallback_navigation_tree(valid_sections, active_slugs)

    tree = _repair_navigation_tree(
        tree.model_copy(
            update={"source_digest": str(facts["source_digest"])}
        ),
        active_slugs,
    )
    tree_errors = _validate_navigation_tree(tree, active_slugs)
    if tree_errors:
        logger.warning(
            "导航树校验失败，改用目录前缀 fallback: %s",
            " | ".join(tree_errors),
        )
        tree = _build_fallback_navigation_tree(valid_sections, active_slugs)
        tree = _repair_navigation_tree(tree, active_slugs)

    # 写入模块正文（路径不变，仍是 modules/*.md）
    await _write_module_files(module_specs, modules_dir, model)

    # 渲染并持久化层级导航文档
    await _atomic_write_text(
        index_path,
        _render_hierarchical_index_md(tree, sections_by_slug),
    )
    await _atomic_write_text(
        _navigation_tree_path(root_path),
        json.dumps(
            tree.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    )
    areas_dir.mkdir(parents=True, exist_ok=True)
    for area in tree.areas:
        await _atomic_write_text(
            areas_dir / f"{area.slug}.md",
            _render_area_md(area, sections_by_slug),
        )

    # 缓存 area_slug 到 registry
    await _persist_navigation_area_cache(root_path, tree)

    # 清理过期文件
    valid_area_slugs = {a.slug for a in tree.areas}
    await _cleanup_stale_area_docs(areas_dir, valid_area_slugs)

    expected_files: set[Path] = {
        Path(ANNOTATIONS_INDEX_FILE),
        Path(NAVIGATION_TREE_FILE),
    }
    expected_files.update(
        Path(ANNOTATIONS_MODULES_DIR) / f"{slug}.md"
        for _, slug, _ in module_specs
    )
    expected_files.update(
        Path(ANNOTATIONS_AREAS_DIR) / f"{a.slug}.md" for a in tree.areas
    )
    await _cleanup_stale_annotation_docs(root_path, expected_files)

    logger.info(
        "树状认知导航写入完成: %d 个区域, %d 个模块文档",
        len(tree.areas),
        len(module_specs),
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
    """增量更新树状导航。

    两档策略：
    - 小改动（ownership 完整 + 受影响模块 ≤3）：局部重写模块文档 + 重渲染导航入口
    - 其余情况：降级为全量 _write_annotations()
    """
    index_path = _annotation_index_path(root_path)
    areas_dir = _annotation_areas_dir(root_path)
    modules_dir = _annotation_modules_dir(root_path)
    entries_map = {str(e.file_meta.path): e for e in entries}

    # 读取现有 navigation_tree.json
    tree_path = _navigation_tree_path(root_path)
    try:
        raw_tree = await asyncio.to_thread(tree_path.read_text, "utf-8")
        tree = NavigationTree.model_validate(json.loads(raw_tree))
    except Exception:
        logger.info(
            "未找到可用 navigation_tree.json，降级为全量认知文档重建"
        )
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    # 从 module_registry 确定 ownership（不再从 index.md 反查）
    manager = ModuleRegistryManager(root_path)
    registry, current_map, historical_map = (
        await manager.build_file_owner_maps()
    )

    affected_slugs: set[str] = set()
    force_full = False

    for fp in changed_files:
        owner = current_map.get(fp)
        if owner is None:
            force_full = True
            break
        affected_slugs.add(owner.slug)

    if not force_full:
        for fp in deleted_files:
            owner = current_map.get(fp) or historical_map.get(fp)
            if owner is None:
                force_full = True
                break
            affected_slugs.add(owner.slug)

    if force_full:
        logger.info("增量 ownership 不完整，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    if not affected_slugs:
        logger.info("认知导航无受影响模块，跳过 annotations 增量更新")
        return

    # 变更范围过大时降级为全量重建
    if len(affected_slugs) > 3:
        logger.info(
            "受影响模块数 %d 超出局部重渲染阈值，降级为全量认知文档重建",
            len(affected_slugs),
        )
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    # 局部更新路径：重写受影响模块文档 + 重渲染导航入口
    recovered_sections = {
        s["slug"]: s
        for s in await _load_sections_from_existing_module_docs(root_path)
    }
    active_records = [
        r for r in registry.records.values() if r.status == "active"
    ]
    sections: list[dict[str, Any]] = []
    for record in active_records:
        file_paths = [
            p for p in _dedupe_keep_order(record.file_paths)
            if p in entries_map
        ]
        if not file_paths:
            continue
        section = recovered_sections.get(record.slug) or {
            "name": record.display_name,
            "slug": record.slug,
            "file_paths": [],
            "body_lines": [
                f"详细认知：.pce/annotations/modules/{record.slug}.md",
            ],
        }
        section["name"] = record.display_name
        _update_section_file_list(section, file_paths)
        _update_section_module_link(section, slug=record.slug)
        sections.append(section)

    sections = _dedupe_sections_by_slug(sections)
    valid_sections, module_specs = _prepare_module_specs(sections, entries_map)
    specs_by_slug = {
        slug: (name, slug, me) for name, slug, me in module_specs
    }

    # 只重写受影响的模块文档
    affected_specs = [
        specs_by_slug[slug]
        for slug in sorted(affected_slugs)
        if slug in specs_by_slug
    ]
    if not affected_specs:
        logger.info("受影响模块无可写入规格，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    await _write_module_files(affected_specs, modules_dir, model)

    # 修复并重渲染导航树
    active_slugs = {slug for _, slug, _ in module_specs}
    tree = _repair_navigation_tree(
        tree.model_copy(update={
            "generated_at": datetime.now(UTC),
            "source_digest": _compute_source_digest(valid_sections),
        }),
        active_slugs,
    )
    tree_errors = _validate_navigation_tree(tree, active_slugs)
    if tree_errors:
        logger.info(
            "现有导航树不再可信，降级为全量认知文档重建: %s",
            " | ".join(tree_errors),
        )
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    sections_by_slug = {s["slug"]: s for s in valid_sections}
    await _atomic_write_text(
        index_path,
        _render_hierarchical_index_md(tree, sections_by_slug),
    )
    areas_dir.mkdir(parents=True, exist_ok=True)
    for area in tree.areas:
        await _atomic_write_text(
            areas_dir / f"{area.slug}.md",
            _render_area_md(area, sections_by_slug),
        )
    await _atomic_write_text(
        tree_path,
        json.dumps(
            tree.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        + "\n",
    )
    await _persist_navigation_area_cache(root_path, tree)
    await _cleanup_stale_area_docs(areas_dir, {a.slug for a in tree.areas})

    expected_files: set[Path] = {
        Path(ANNOTATIONS_INDEX_FILE),
        Path(NAVIGATION_TREE_FILE),
    }
    expected_files.update(
        Path(ANNOTATIONS_MODULES_DIR) / f"{slug}.md"
        for _, slug, _ in module_specs
    )
    expected_files.update(
        Path(ANNOTATIONS_AREAS_DIR) / f"{a.slug}.md" for a in tree.areas
    )
    await _cleanup_stale_annotation_docs(root_path, expected_files)

    logger.info(
        "认知导航增量更新完成: %d 个模块重建, %d 个区域重渲染",
        len(affected_specs),
        len(tree.areas),
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
        await _ensure_generated_pceignore(root_path, serena_client, model=model)
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
        await _write_annotations(entries, project_meta, memory_root_path, model=model)
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
