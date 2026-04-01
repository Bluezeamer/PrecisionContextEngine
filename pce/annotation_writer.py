"""PCE 认知文档生成模块。

负责生成和维护三层认知导航文档：
1. annotations/index.md — 项目级认知导航入口
2. annotations/areas/*.md — 区域级导航文档
3. annotations/modules/*.md — 模块深度认知文档
4. structure.md — 项目结构导航
5. pceignore — 文件过滤规则

从 indexer.py 分离而来，indexer 只保留文件发现和符号索引构建。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import tomllib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from ._env import build_litellm_model, get_completion_overrides, get_env_text, get_temperature
from .memory import (
    ANNOTATIONS_DIR,
    ANNOTATIONS_AREAS_DIR,
    ANNOTATIONS_MODULES_DIR,
    NAVIGATION_TREE_FILE,
    _atomic_write_text,
    load_file_baseline,
    load_index,
)
from .module_annotation_contract import validate_module_annotation_markdown
from .models import (
    AreaRecord,
    IndexEntry,
    ModuleCognitionFacts,
    ModuleNavRecord,
    ModuleRegistry,
    NavigationTree,
    ProjectMeta,
)
from .module_registry import ModuleRegistryManager
from .prompt_guard import build_prompt_budget, estimate_input_tokens, fit_text_to_budget
from .serena_client import SerenaClient, SerenaClientError
from .topology_cognition_agent import TopologyCognitionAgent

logger = logging.getLogger(__name__)

# 常量（与 indexer.py 共享的导航路径常量）
ANNOTATIONS_INDEX_FILE = "index.md"
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
_ANNOTATION_LLM_TIMEOUT_SECONDS = 20.0
_MODULE_ANNOTATION_CONCURRENCY = 4
_INDEX_PROMPT_MAX_SUMMARY_LINES = 48
_MODULE_PROMPT_MAX_FILE_LINES = 24
_MODULE_PROMPT_SYMBOL_ANCHORS_PER_FILE = 2
_MODULE_PROMPT_SYMBOL_ANCHORS_TOTAL = 6
_MODULE_FALLBACK_SYMBOL_ANCHORS_TOTAL = 4
_MISSING_COVERAGE_INDEX_BUDGET = 5000


def _normalize_tool_result(value: Any) -> Any:
    """将工具返回值统一化为 dict/list 结构。

    Serena 工具返回值经过 _jsonable 处理后可能有多种形态：
    1. 直接返回 dict/list（最理想）
    2. 单元素列表包含 JSON 字符串（旧版兼容）
    3. {'meta': ..., 'content': [{'type': 'text', 'text': <data>}]} 外壳（实测形态）
    """
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
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
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        try:
            return json.loads(value[0])
        except (ValueError, json.JSONDecodeError):
            return value[0]
    return value


def _annotation_model_for_budget() -> str | None:
    return _resolve_annotation_model(None)


def _fit_lines_with_budget(
    *,
    prefix_lines: list[str],
    candidate_lines: list[str],
    suffix_lines: list[str],
    max_items: int,
    min_items: int,
) -> list[str]:
    model = _annotation_model_for_budget()
    if model is None:
        return candidate_lines[:max_items]

    budget = build_prompt_budget()
    selected = candidate_lines[:max_items]
    while len(selected) > min_items:
        prompt = "\n".join([*prefix_lines, *selected, *suffix_lines])
        tokens = estimate_input_tokens(
            model,
            [
                {"role": "system", "content": "annotation-writer-budget-check"},
                {"role": "user", "content": prompt},
            ],
        )
        if tokens <= budget.target_input_budget:
            return selected
        selected = selected[:-4]
    return selected[:min_items]

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


def _topology_debug_dir(root_path: Path) -> Path:
    """返回 topology init 调试输出目录。"""
    return _annotation_root_dir(root_path) / "_topology_debug"


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


def _format_exception_brief(exc: BaseException) -> str:
    """将异常格式化为稳定、非空的简短字符串。"""
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__}: {exc!r}"


async def _persist_topology_stage_debug(
    root_path: Path,
    *,
    stage: str,
    payload: Any,
    extra: dict[str, Any] | None = None,
) -> None:
    """持久化 topology init 各阶段原始输出，便于排查结构漂移。"""
    debug_dir = _topology_debug_dir(root_path)
    debug_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "recorded_at": datetime.now(UTC).isoformat(),
        "payload": payload,
    }
    if extra:
        record["extra"] = extra
    await _atomic_write_text(
        debug_dir / f"{stage}.json",
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    )


async def _persist_topology_debug_error(
    root_path: Path,
    *,
    stage: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """持久化 topology init 阶段错误。"""
    debug_dir = _topology_debug_dir(root_path)
    debug_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "recorded_at": datetime.now(UTC).isoformat(),
        "error": message,
    }
    if extra:
        record["extra"] = extra
    await _atomic_write_text(
        debug_dir / f"{stage}_error.json",
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    )


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
    budget = build_prompt_budget()
    tokens = estimate_input_tokens(effective_model, messages)
    if tokens > budget.soft_input_budget:
        trimmed_prompt = fit_text_to_budget(
            effective_model,
            prompt,
            token_budget=max(800, budget.target_input_budget // 2),
            notice="\n\n[提示词过长，已保留前后关键部分]\n\n",
        )
        messages[1]["content"] = trimmed_prompt
        tokens = estimate_input_tokens(effective_model, messages)
        if tokens > budget.hard_input_budget:
            trimmed_prompt = fit_text_to_budget(
                effective_model,
                trimmed_prompt,
                token_budget=max(600, budget.target_input_budget // 3),
                notice="\n\n[提示词进一步压缩]\n\n",
            )
            messages[1]["content"] = trimmed_prompt
        logger.warning("%s: prompt_guard 生效，tokens=%d", failure_log, tokens)
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                litellm.completion,
                model=effective_model,
                messages=messages,
                temperature=get_temperature(
                    specific_key="PCE_ANNOTATION_TEMPERATURE",
                    default=0.1,
                ),
                **get_completion_overrides(),
            ),
            timeout=_ANNOTATION_LLM_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logger.warning("%s(已降级): %s", failure_log, _format_exception_brief(e))
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
    prefix_lines = [
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
    ]
    suffix_lines = [
        "",
        "输出约束:",
        "- 第一行必须是 `# 项目认知导航`",
        "- 每个模块使用 `## 模块名` 开头",
        "- `文件：` 行使用逗号分隔的相对路径，路径必须来自索引摘要",
        "- `职责：` 2-3 句话，描述模块边界、关键职责、主要协作对象和高风险点",
        "- `详细认知：` 写成 `.pce/annotations/modules/{slug}.md`，slug = 模块名小写后空格替换为连字符",
        "- 不要输出 JSON、代码块、前言、结语或任何未在示例中出现的附加文本",
    ]
    raw_summary_lines = [_format_entry_outline(entry) for entry in entries[:80]]
    summary_lines = _fit_lines_with_budget(
        prefix_lines=prefix_lines,
        candidate_lines=raw_summary_lines,
        suffix_lines=suffix_lines,
        max_items=_INDEX_PROMPT_MAX_SUMMARY_LINES,
        min_items=12,
    )
    return "\n".join([*prefix_lines, *summary_lines, *suffix_lines])


def _build_module_annotation_prompt(
    module_name: str,
    module_entries: list[IndexEntry],
) -> str:
    """构建单模块深度认知文档的 LLM 提示词。"""
    raw_file_lines: list[str] = []
    remaining_symbol_anchors = _MODULE_PROMPT_SYMBOL_ANCHORS_TOTAL
    for entry in module_entries:
        anchor_limit = min(_MODULE_PROMPT_SYMBOL_ANCHORS_PER_FILE, remaining_symbol_anchors)
        anchor_lines = [
            f"{sym.name}({sym.kind.value})@{sym.line_start}"
            for sym in entry.symbols[:anchor_limit]
        ]
        remaining_symbol_anchors = max(0, remaining_symbol_anchors - len(anchor_lines))
        line = f"- {entry.file_meta.path} [{entry.file_meta.language}, {entry.file_meta.loc} 行]"
        if anchor_lines:
            line += f" 锚点: {', '.join(anchor_lines)}"
            if len(entry.symbols) > anchor_limit:
                line += " ..."
        raw_file_lines.append(line)

    prefix_lines = [
        "你要为单个代码模块生成可按需加载的深度认知文档。",
        "请输出 Markdown，不要代码块，不要额外解释。",
        "",
        f"模块名: {module_name}",
        f"覆盖文件数: {len(module_entries)}",
        "",
        "文件与符号摘要:",
    ]
    suffix_lines = [
        "",
        "输出要求:",
        f"- 标题必须是 `# {module_name}`",
        "- 必须包含 `## 覆盖文件`、`## 核心职责`、`## 关键流程`、`## 外部协作`、`## 风险与约束` 这些二级标题",
        "- `## 覆盖文件` 需列出所有文件路径",
        "- 如确有必要，可额外写 `## 关键符号`，但只能保留少量高价值锚点，禁止枚举实现细节",
        "- `## 关键流程` 关注控制流、数据流或调用链，不要泛泛而谈",
        "- 结论必须基于输入，不得编造不存在的文件、符号或依赖",
    ]
    file_lines = _fit_lines_with_budget(
        prefix_lines=prefix_lines,
        candidate_lines=raw_file_lines,
        suffix_lines=suffix_lines,
        max_items=_MODULE_PROMPT_MAX_FILE_LINES,
        min_items=4,
    )
    return "\n".join([*prefix_lines, *file_lines, *suffix_lines])


def _build_module_assignment_prompt(entry: IndexEntry, index_content: str) -> str:
    """构建新增文件模块归属判断的 LLM 提示词。"""
    model = _annotation_model_for_budget()
    compact_index = index_content.strip()
    if model is not None:
        compact_index = fit_text_to_budget(
            model,
            compact_index,
            token_budget=_MISSING_COVERAGE_INDEX_BUDGET,
            notice="\n\n[现有导航过长，已压缩]\n\n",
            min_chars=1200,
        )
    return "\n".join(
        [
            "请根据现有项目认知导航，为一个新增文件判断最合适的模块归属。",
            "只能从已有模块 slug 中选择；如果无法可靠判断，则返回空字符串。",
            "",
            "现有导航:",
            compact_index,
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
    model = _annotation_model_for_budget()
    compact_index = index_content.strip()
    if model is not None:
        compact_index = fit_text_to_budget(
            model,
            compact_index,
            token_budget=_MISSING_COVERAGE_INDEX_BUDGET,
            notice="\n\n[现有导航过长，已压缩]\n\n",
            min_chars=1500,
        )
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
            compact_index,
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

    lines.extend(["", "## 核心职责", "当前为基于索引条目的回退摘要，提供基本文件边界与少量入口锚点。"])

    anchors: list[str] = []
    for entry in module_entries:
        for sym in entry.symbols[:1]:
            anchors.append(f"- {sym.name} ({sym.kind.value}) — {sym.file_path}:{sym.line_start}")
            if len(anchors) >= _MODULE_FALLBACK_SYMBOL_ANCHORS_TOTAL:
                break
        if len(anchors) >= _MODULE_FALLBACK_SYMBOL_ANCHORS_TOTAL:
            break
    if anchors:
        lines.extend(["", "## 专题：关键锚点", *anchors])

    lines.extend(
        [
            "",
            "## 关键流程",
            "需结合具体文件内容进一步检索，当前回退摘要不直接推断控制流；符号细节请优先通过 Serena 工具动态查询。",
            "",
            "## 外部协作",
            "可结合 structure.md、references.json 及 Serena 工具继续定位模块交互关系。",
            "",
            "## 风险与约束",
            "本文件由索引级回退逻辑生成，细粒度职责与边界仍应以 LLM 生成版本为准。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_empty_module_skeleton_md(
    module_name: str,
    module_entries: list[IndexEntry],
) -> str:
    lines = [
        f"# {module_name}",
        "",
        "## 覆盖文件",
    ]
    for entry in module_entries:
        lines.append(f"- {entry.file_meta.path}")
    lines.extend(
        [
            "",
            "## 核心职责",
            "",
            "## 关键流程",
            "",
            "## 外部协作",
            "",
            "## 风险与约束",
            "",
            "## 关键符号",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


async def _compute_file_hash(path: Path) -> str | None:
    def _hash() -> str | None:
        if not path.exists() or not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    return await asyncio.to_thread(_hash)


def _rewrite_coverage_section(content: str, file_paths: list[str], module_name: str) -> str:
    lines = content.splitlines()
    title = module_name
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def _flush() -> None:
        nonlocal current_heading, current_body
        if current_heading is None:
            return
        sections.append((current_heading, current_body[:]))

    for line in lines:
        if line.startswith("# ") and title == module_name:
            title = line[2:].strip() or module_name
            continue
        if line.startswith("## "):
            _flush()
            current_heading = line[3:].strip()
            current_body = []
            continue
        if current_heading is not None:
            current_body.append(line.rstrip())

    _flush()

    replaced = False
    rendered = [f"# {title}", "", "## 覆盖文件", *[f"- {path}" for path in file_paths]]
    for heading, body in sections:
        if heading == "覆盖文件":
            replaced = True
            continue
        rendered.extend(["", f"## {heading}", *body])
    if not replaced and not sections:
        rendered.extend(["", "## 核心职责"])
    return "\n".join(rendered).rstrip() + "\n"


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
        residual_count = sum(
            1 for module in area.modules if module.module_type == "residual"
        )
        if residual_count > 1:
            errors.append(f"区域 {area.slug} residual 模块超过 1 个")

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
    fallback_modules: list[ModuleNavRecord] = []
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
        kept_modules: list[ModuleNavRecord] = []
        for module in area.modules:
            if module.slug not in active or module.slug in used_modules:
                continue
            used_modules.add(module.slug)
            kept_modules.append(module)
        mods = [module.slug for module in kept_modules]

        order = [
            s for s in _dedupe_keep_order(area.recommended_order)
            if s in set(mods)
        ]
        for s in mods:
            if s not in order:
                order.append(s)

        if area.is_fallback:
            fallback_modules.extend(kept_modules)
            fallback_order.extend(order)
            fallback_prefixes.extend(area.source_prefixes)
            fallback_summary = fallback_summary or area.summary
            continue

        if not mods:
            continue

        repaired_areas.append(area.model_copy(update={
            "slug": _uniq_slug(area.slug),
            "modules": kept_modules,
            "module_slugs": mods,
            "recommended_order": order,
        }))

    # 缺失模块塞入 fallback
    missing = [s for s in sorted(active) if s not in used_modules]
    fallback_modules_by_slug = {
        module.slug: module for module in fallback_modules
    }
    for area in tree.areas:
        for module in area.modules:
            if module.slug in missing and module.slug not in fallback_modules_by_slug:
                fallback_modules_by_slug[module.slug] = module
    fallback_modules = list(fallback_modules_by_slug.values())
    fallback_order = _dedupe_keep_order([*fallback_order, *[module.slug for module in fallback_modules]])

    # area 数量上限 7（+1 fallback = 8）
    if len(repaired_areas) > 7:
        for overflow in repaired_areas[7:]:
            fallback_modules.extend(overflow.modules)
            fallback_order.extend(overflow.recommended_order)
        repaired_areas = repaired_areas[:7]
        fallback_modules = list({
            module.slug: module for module in fallback_modules
        }.values())
        fallback_order = _dedupe_keep_order(
            [*fallback_order, *[module.slug for module in fallback_modules]]
        )

    # 至少保留 1 个非 fallback area，避免模型把所有模块都塞进 fallback
    if not repaired_areas and fallback_modules:
        logger.warning(
            "navigation tree repair: 检测到 fallback-only 结构，自动生成默认主区域 core"
        )
        seed_modules = fallback_modules[: min(len(fallback_modules), 8)]
        seed_slugs = [module.slug for module in seed_modules]
        repaired_areas.append(AreaRecord(
            slug=_uniq_slug("core"),
            display_name="核心模块",
            summary="模型未给出稳定区域划分时的默认主区域。",
            modules=seed_modules,
            module_slugs=seed_slugs,
            recommended_order=seed_slugs,
            source_prefixes=[],
            is_fallback=False,
        ))
        fallback_modules = [
            module for module in fallback_modules if module.slug not in set(seed_slugs)
        ]
        fallback_order = [slug for slug in fallback_order if slug not in set(seed_slugs)]

    fb_slug = tree.fallback_area_slug or NAVIGATION_FALLBACK_AREA_SLUG
    if fb_slug in used_area_slugs:
        fb_slug = _uniq_slug(NAVIGATION_FALLBACK_AREA_SLUG)
    else:
        used_area_slugs.add(fb_slug)

    fb_area = AreaRecord(
        slug=fb_slug,
        display_name=NAVIGATION_FALLBACK_AREA_NAME,
        summary=fallback_summary or "承接暂时无法稳定归入主区域的模块。",
        modules=fallback_modules,
        module_slugs=[module.slug for module in fallback_modules],
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


def _module_nav_record_by_slug(area: AreaRecord, slug: str) -> Any | None:
    for module in area.modules:
        if module.slug == slug:
            return module
    return None


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
        previews: list[str] = []
        for slug in area.module_slugs[:3]:
            section = sections_by_slug.get(slug, {})
            module = _module_nav_record_by_slug(area, slug)
            previews.append(
                str(
                    section.get("name")
                    or getattr(module, "display_name", None)
                    or slug
                )
            )
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
            module = _module_nav_record_by_slug(area, slug)
            name = str(
                section.get("name")
                or getattr(module, "display_name", None)
                or slug
            )
            summary = (
                _extract_section_summary(section)
                or str(getattr(module, "summary", "") or "").strip()
                or "暂无模块摘要。"
            )
            lines.append(
                f"- [{name}](../modules/{slug}.md) (`{slug}`)：{summary}"
            )
    else:
        lines.append("- 当前无挂载模块。")

    lines.extend(["", "## 推荐阅读顺序"])
    ordered = area.recommended_order or area.module_slugs
    if ordered:
        for idx, slug in enumerate(ordered, start=1):
            section = sections_by_slug.get(slug, {})
            module = _module_nav_record_by_slug(area, slug)
            name = str(
                section.get("name")
                or getattr(module, "display_name", None)
                or slug
            )
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
        cloned = dict(section)
        cloned["file_paths"] = list(section.get("file_paths", []))
        cloned["body_lines"] = list(section.get("body_lines", []))
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            stabilized.append(cloned)
            continue
        record = await manager.get_or_create_module(
            display_name=section["name"],
            file_paths=file_paths,
            key_symbols=_extract_module_key_symbols([entries_map[p] for p in file_paths]),
        )
        cloned["name"] = record.display_name
        _update_section_file_list(cloned, file_paths)
        _update_section_module_link(cloned, slug=record.slug)
        stabilized.append(cloned)
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
    """生成并写入模块认知文档，限制并发避免冷启动压垮模型侧队列。"""
    semaphore = asyncio.Semaphore(_MODULE_ANNOTATION_CONCURRENCY)

    async def _generate_with_limit(
        module_name: str,
        module_entries: list[IndexEntry],
    ) -> str | BaseException | None:
        async with semaphore:
            try:
                return await _generate_module_annotation(
                    module_name,
                    module_entries,
                    model=model,
                )
            except BaseException as exc:  # noqa: BLE001
                return exc

    results = await asyncio.gather(
        *[
            _generate_with_limit(name, entries)
            for name, _, entries in module_specs
        ],
        return_exceptions=False,
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


def _build_topology_discovery_facts(
    entries_map: dict[str, IndexEntry],
    root_path: Path,
) -> dict[str, Any]:
    """构建 topology init 的最小 files_tree facts。"""
    tree_root: dict[str, Any] = {"children": {}}

    for entry in sorted(entries_map.values(), key=lambda item: str(item.file_meta.path)):
        path_str = str(entry.file_meta.path)
        parts = Path(path_str).parts
        if not parts:
            continue

        cursor = tree_root["children"]
        prefix_parts: list[str] = []
        for part in parts[:-1]:
            prefix_parts.append(part)
            dir_path = "/".join(prefix_parts) + "/"
            cursor = cursor.setdefault(dir_path, {"path": dir_path, "children": {}})["children"]

        file_path = "/".join(parts)
        cursor[file_path] = {
            "path": file_path,
            "language": entry.file_meta.language,
        }

    def _finalize_children(children: dict[str, Any]) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for key in sorted(children.keys()):
            node = children[key]
            raw_children = node.get("children")
            if isinstance(raw_children, dict):
                finalized.append({
                    "path": node["path"],
                    "children": _finalize_children(raw_children),
                })
            else:
                finalized.append({
                    "path": node["path"],
                    "language": node["language"],
                })
        return finalized

    return {
        "project_root": root_path.name,
        "files_tree": _finalize_children(tree_root["children"]),
    }


def _build_topology_missing_paths_tree(paths: list[str], entries_map: dict[str, IndexEntry]) -> list[dict[str, Any]]:
    """为 navigation repair 阶段构建未挂载文件树。"""
    tree_root: dict[str, Any] = {"children": {}}

    for path_str in sorted({path for path in paths if path in entries_map}):
        parts = Path(path_str).parts
        if not parts:
            continue

        cursor = tree_root["children"]
        prefix_parts: list[str] = []
        for part in parts[:-1]:
            prefix_parts.append(part)
            dir_path = "/".join(prefix_parts) + "/"
            cursor = cursor.setdefault(dir_path, {"path": dir_path, "children": {}})["children"]

        cursor[path_str] = {
            "path": path_str,
            "language": entries_map[path_str].file_meta.language,
        }

    def _finalize_children(children: dict[str, Any]) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        for key in sorted(children.keys()):
            node = children[key]
            raw_children = node.get("children")
            if isinstance(raw_children, dict):
                finalized.append({
                    "path": node["path"],
                    "children": _finalize_children(raw_children),
                })
            else:
                finalized.append({
                    "path": node["path"],
                    "language": node["language"],
                })
        return finalized

    return _finalize_children(tree_root["children"])


def _collect_navigation_tree_coverage_issues(
    sections: list[dict[str, Any]],
    entries_map: dict[str, IndexEntry],
) -> dict[str, Any]:
    assigned_counts: Counter[str] = Counter()
    invalid_paths: list[str] = []

    for section in sections:
        for file_path in _dedupe_keep_order(section.get("file_paths", [])):
            if file_path in entries_map:
                assigned_counts[file_path] += 1
            else:
                invalid_paths.append(file_path)

    unassigned_paths = sorted(path for path in entries_map if assigned_counts[path] == 0)
    duplicate_paths = sorted(path for path, count in assigned_counts.items() if count > 1)
    return {
        "unassigned_paths": unassigned_paths,
        "duplicate_paths": duplicate_paths,
        "invalid_paths": sorted(set(invalid_paths)),
    }


def _collect_navigation_payload_issues(
    tree: NavigationTree,
    entries_map: dict[str, IndexEntry],
) -> dict[str, Any]:
    """从 navigation_tree 的最终归属结果收集覆盖问题。"""
    assignment = _assign_navigation_tree_paths(tree, entries_map)
    return {
        "unassigned_paths": list(assignment["unassigned_paths"]),
        "duplicate_paths": list(assignment["duplicate_paths"]),
        "duplicate_details": list(assignment["duplicate_details"]),
        "invalid_paths": list(assignment["invalid_paths"]),
        "area_residuals": list(assignment["area_residuals"]),
        "too_many_file_centered_modules": list(
            assignment["too_many_file_centered_modules"]
        ),
    }


def _build_navigation_repair_feedback(
    *,
    attempt: int,
    max_attempts: int,
    issues: dict[str, Any],
) -> str:
    unassigned_paths = list(issues.get("unassigned_paths") or [])
    duplicate_paths = list(issues.get("duplicate_paths") or [])
    duplicate_details = list(issues.get("duplicate_details") or [])
    invalid_paths = list(issues.get("invalid_paths") or [])
    lines = [
        f"上一轮 `navigation_tree` 存在覆盖问题，需要补齐（第 {attempt}/{max_attempts} 次 repair）。",
        "请保留已有结构中合理的 area/module 划分，仅对遗漏或错误挂载做最小修正。",
    ]
    if unassigned_paths:
        lines.append(f"- 未挂载文件数：{len(unassigned_paths)}")
    if duplicate_paths:
        lines.append(f"- 重复挂载文件数：{len(duplicate_paths)}")
    if duplicate_details:
        lines.append("- 典型重复挂载：")
        for item in duplicate_details[:8]:
            path = str(item.get("path") or "").strip()
            modules = ", ".join(str(x).strip() for x in item.get("modules") or [] if str(x).strip())
            if path and modules:
                lines.append(f"  - {path}: {modules}")
    if invalid_paths:
        lines.append(f"- 非法路径数：{len(invalid_paths)}")
    area_residuals = list(issues.get("area_residuals") or [])
    if area_residuals:
        lines.append("- area 内仍存在较大 residual：")
        for item in area_residuals[:6]:
            area_slug = str(item.get("area_slug") or "").strip()
            count = int(item.get("residual_count") or 0)
            samples = ", ".join(
                str(x).strip() for x in item.get("sample_paths") or [] if str(x).strip()
            )
            if area_slug and count > 0:
                suffix = f"（样例：{samples}）" if samples else ""
                lines.append(f"  - {area_slug}: {count} 个{suffix}")
    too_many_file_centered = list(issues.get("too_many_file_centered_modules") or [])
    if too_many_file_centered:
        lines.append("- 某些 area 的 file_centered 模块过多，应优先合并为目录模块或 residual：")
        for item in too_many_file_centered[:6]:
            area_slug = str(item.get("area_slug") or "").strip()
            count = int(item.get("count") or 0)
            if area_slug and count > 0:
                lines.append(f"  - {area_slug}: {count} 个")
    lines.extend([
        "优先按目录级边界修正 area/module；单文件 module 只在平铺结构且确有中心职责时保留。",
        "优先把遗漏文件吸收到已有目录模块或 area residual；仅在明显需要时新增少量 module。",
        "对每一个重复挂载文件，必须强行选择唯一 module 归属；不能保留模糊共存或双归属。",
        "若某个宽泛 include 与更精确的模块冲突，请保留精确模块，并给宽泛模块增加 exclude，直到冲突文件只剩唯一归属。",
        "若本轮仍有一批挂载不明朗的文件，请批量处理这些冲突文件的唯一归属，而不是回避或仅部分修正。",
        "修正后请重新提交完整 `navigation_tree`。",
    ])
    return "\n".join(lines)


def _expand_module_path_rules(
    module: ModuleNavRecord,
    candidate_paths: set[str],
) -> tuple[list[str], list[str]]:
    """按 glob 规则展开模块覆盖范围。"""
    include_patterns = _dedupe_keep_order(list(module.include))
    exclude_patterns = _dedupe_keep_order(list(module.exclude))
    invalid_patterns: list[str] = []
    matched: list[str] = []

    for pattern in include_patterns:
        normalized = str(pattern).strip()
        if not normalized or normalized.startswith("!"):
            invalid_patterns.append(normalized)
            continue
        matched.extend(
            path for path in sorted(candidate_paths)
            if Path(path).match(normalized)
        )

    resolved = _dedupe_keep_order(matched)
    if exclude_patterns:
        excluded: set[str] = set()
        for pattern in exclude_patterns:
            normalized = str(pattern).strip()
            if not normalized or normalized.startswith("!"):
                invalid_patterns.append(normalized)
                continue
            excluded.update(
                path for path in resolved
                if Path(path).match(normalized)
            )
        resolved = [path for path in resolved if path not in excluded]

    return resolved, [pattern for pattern in invalid_patterns if pattern]


def _normalize_module_nav_type(value: Any) -> str:
    text = " ".join(str(value or "").strip().split()).lower()
    if text in {"directory", "dir"}:
        return "directory"
    if text in {"residual", "fallback", "remainder"}:
        return "residual"
    return "file_centered"


def _merge_navigation_module_payloads(
    modules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged_by_slug: dict[str, dict[str, Any]] = {}
    merged_order: list[str] = []
    type_rank = {"file_centered": 0, "directory": 1, "residual": 2}

    for module in modules:
        slug = str(module.get("slug") or "").strip()
        if not slug:
            continue
        existing = merged_by_slug.get(slug)
        if existing is None:
            merged_by_slug[slug] = {
                "slug": slug,
                "display_name": module["display_name"],
                "summary": module.get("summary", ""),
                "module_type": module["module_type"],
                "include": list(module.get("include", [])),
                "exclude": list(module.get("exclude", [])),
            }
            merged_order.append(slug)
            continue

        if not existing.get("summary") and module.get("summary"):
            existing["summary"] = module["summary"]
        existing["include"] = _dedupe_keep_order(
            list(existing.get("include", [])) + list(module.get("include", []))
        )
        existing["exclude"] = _dedupe_keep_order(
            list(existing.get("exclude", [])) + list(module.get("exclude", []))
        )
        if type_rank[module["module_type"]] > type_rank[existing["module_type"]]:
            existing["module_type"] = module["module_type"]

    return [merged_by_slug[slug] for slug in merged_order]


def _extract_rule_prefix(pattern: str) -> str:
    normalized = str(pattern).strip()
    if not normalized:
        return ""
    wildcard_positions = [
        pos for pos in (
            normalized.find("*"),
            normalized.find("?"),
            normalized.find("["),
        ) if pos >= 0
    ]
    if wildcard_positions:
        normalized = normalized[: min(wildcard_positions)]
    normalized = normalized.rstrip("/")
    if "/" not in normalized:
        return ""
    if normalized.endswith((".", "-", "_")):
        normalized = normalized[:-1]
    if "/" in normalized:
        normalized = normalized.rsplit("/", 1)[0]
    return normalized.strip("/")


def _directory_roots_for_module(module: ModuleNavRecord) -> list[str]:
    if module.module_type != "directory":
        return []
    roots = [
        _extract_rule_prefix(pattern)
        for pattern in module.include
    ]
    return [root for root in _dedupe_keep_order(roots) if root]


def _path_matches_prefix(path: str, prefix: str) -> bool:
    normalized_prefix = str(prefix).strip().strip("/")
    if not normalized_prefix:
        return True
    return path == normalized_prefix or path.startswith(normalized_prefix + "/")


def _path_specificity(path: str, module: ModuleNavRecord) -> tuple[int, int, int]:
    resolved_paths, _ = _expand_module_path_rules(module, {path})
    explicit_hit = 1 if resolved_paths else 0
    roots = _directory_roots_for_module(module)
    best_root = max((len(root) for root in roots if _path_matches_prefix(path, root)), default=0)
    return (explicit_hit, best_root, len(module.include))


def _derive_area_candidate_paths(
    area: AreaRecord,
    *,
    candidate_paths: set[str],
    explicit_module_paths: dict[str, list[str]],
) -> set[str]:
    derived: set[str] = set()
    for prefix in area.source_prefixes:
        derived.update(path for path in candidate_paths if _path_matches_prefix(path, prefix))

    for module in area.modules:
        for root in _directory_roots_for_module(module):
            derived.update(path for path in candidate_paths if _path_matches_prefix(path, root))
        derived.update(explicit_module_paths.get(module.slug, []))
    return derived


def _assign_navigation_tree_paths(
    tree: NavigationTree,
    entries_map: dict[str, IndexEntry],
) -> dict[str, Any]:
    candidate_paths = set(entries_map)
    module_by_slug: dict[str, ModuleNavRecord] = {}
    module_area_by_slug: dict[str, str] = {}
    explicit_module_paths: dict[str, list[str]] = {}
    owners: dict[str, list[str]] = {}
    invalid_paths: list[str] = []
    file_centered_counter: Counter[str] = Counter()

    for area in tree.areas:
        for module in area.modules:
            module_by_slug[module.slug] = module
            module_area_by_slug[module.slug] = area.slug
            if module.module_type == "file_centered":
                file_centered_counter[area.slug] += 1
            resolved_paths: list[str] = []
            rule_errors: list[str] = []
            if module.module_type != "residual" and module.include:
                resolved_paths, rule_errors = _expand_module_path_rules(module, candidate_paths)
            invalid_paths.extend(f"{module.slug}:{item}" for item in rule_errors)
            explicit_module_paths[module.slug] = resolved_paths
            for file_path in resolved_paths:
                owners.setdefault(file_path, []).append(module.slug)

    duplicate_paths = sorted(path for path, slugs in owners.items() if len(set(slugs)) > 1)
    duplicate_details = [
        {"path": path, "modules": _dedupe_keep_order(owners.get(path, []))}
        for path in duplicate_paths
    ]

    final_owner: dict[str, str] = {}
    module_paths: dict[str, list[str]] = {slug: [] for slug in module_by_slug}

    for path, slugs in owners.items():
        unique_slugs = _dedupe_keep_order(slugs)
        if len(unique_slugs) != 1:
            continue
        final_owner[path] = unique_slugs[0]
        module_paths.setdefault(unique_slugs[0], []).append(path)

    area_paths: dict[str, set[str]] = {}
    for area in tree.areas:
        area_paths[area.slug] = _derive_area_candidate_paths(
            area,
            candidate_paths=candidate_paths,
            explicit_module_paths=explicit_module_paths,
        )

    for area in tree.areas:
        if area.is_fallback:
            continue

        residual_module = next(
            (module for module in area.modules if module.module_type == "residual"),
            None,
        )
        directory_modules = [
            module for module in area.modules if module.module_type == "directory"
        ]
        area_candidates = area_paths.get(area.slug, set())
        if not area_candidates:
            continue

        for path in sorted(area_candidates):
            if path in final_owner:
                continue
            directory_hits = [
                module for module in directory_modules
                if any(_path_matches_prefix(path, root) for root in _directory_roots_for_module(module))
            ]
            if len(directory_hits) == 1:
                owner = directory_hits[0].slug
                final_owner[path] = owner
                module_paths.setdefault(owner, []).append(path)
                continue
            if len(directory_hits) > 1:
                ranked = sorted(
                    directory_hits,
                    key=lambda module: _path_specificity(path, module),
                    reverse=True,
                )
                if len(ranked) == 1 or _path_specificity(path, ranked[0]) > _path_specificity(path, ranked[1]):
                    owner = ranked[0].slug
                    final_owner[path] = owner
                    module_paths.setdefault(owner, []).append(path)
                    continue
            if residual_module is not None:
                owner = residual_module.slug
                final_owner[path] = owner
                module_paths.setdefault(owner, []).append(path)

    fallback_area = next((area for area in tree.areas if area.is_fallback), None)
    if fallback_area is not None:
        fallback_residual = next(
            (module for module in fallback_area.modules if module.module_type == "residual"),
            None,
        )
        for path in sorted(candidate_paths):
            if path in final_owner:
                continue
            if fallback_residual is not None:
                owner = fallback_residual.slug
                final_owner[path] = owner
                module_paths.setdefault(owner, []).append(path)

    unassigned_paths = sorted(path for path in candidate_paths if path not in final_owner)
    area_residuals: list[dict[str, Any]] = []
    too_many_file_centered_modules: list[dict[str, Any]] = []

    for area in tree.areas:
        residual_modules = [
            module for module in area.modules if module.module_type == "residual"
        ]
        if len(residual_modules) > 1:
            invalid_paths.append(f"{area.slug}:multiple_residual_modules")
        if file_centered_counter[area.slug] > 4:
            too_many_file_centered_modules.append({
                "area_slug": area.slug,
                "count": file_centered_counter[area.slug],
            })
        residual_module = residual_modules[0] if residual_modules else None
        if residual_module is None:
            continue
        residual_paths = sorted(module_paths.get(residual_module.slug, []))
        if len(residual_paths) >= 3:
            area_residuals.append({
                "area_slug": area.slug,
                "residual_slug": residual_module.slug,
                "residual_count": len(residual_paths),
                "sample_paths": residual_paths[:8],
            })

    for slug, paths in list(module_paths.items()):
        module_paths[slug] = sorted(_dedupe_keep_order(paths))

    return {
        "module_paths": module_paths,
        "path_owner": final_owner,
        "unassigned_paths": unassigned_paths,
        "duplicate_paths": duplicate_paths,
        "duplicate_details": duplicate_details,
        "invalid_paths": sorted(set(invalid_paths)),
        "area_residuals": area_residuals,
        "too_many_file_centered_modules": too_many_file_centered_modules,
    }


def _strip_markdown_emphasis(text: str) -> str:
    return re.sub(r"[*_`]+", "", text).strip()


def _module_nav_summary_from_cognition_fact(fact: Any) -> str:
    candidates = [
        *(list(getattr(fact, "core_responsibility", []) or [])),
        *(list(getattr(fact, "key_flow", []) or [])),
        *(list(getattr(fact, "external_collaboration", []) or [])),
        *(list(getattr(fact, "risks_constraints", []) or [])),
    ]
    for item in candidates:
        text = _strip_markdown_emphasis(" ".join(str(item).strip().split()))
        if text:
            return text
    return ""


def _truncate_sentence(text: str, *, max_chars: int) -> str:
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _is_generic_project_summary(summary: str) -> bool:
    text = " ".join(summary.strip().split())
    if not text:
        return True
    return text.startswith("项目当前共 ") or text.startswith("共 ")


def _is_generic_area_summary(summary: str) -> bool:
    text = " ".join(summary.strip().split())
    if not text:
        return True
    return (
        text.startswith("主要覆盖 `")
        or text.startswith("承接")
        or text.startswith("覆盖 `")
        or text.startswith("模型未给出稳定区域划分时")
    )


def _inject_section_summaries_from_cognition(
    sections: list[dict[str, Any]],
    cognition_facts: ModuleCognitionFacts,
) -> list[dict[str, Any]]:
    facts_by_slug = {fact.slug: fact for fact in cognition_facts.modules}
    injected: list[dict[str, Any]] = []
    for section in sections:
        cloned = {
            "name": section["name"],
            "slug": section["slug"],
            "file_paths": list(section.get("file_paths", [])),
            "body_lines": list(section.get("body_lines", [])),
        }
        summary = _module_nav_summary_from_cognition_fact(facts_by_slug.get(section["slug"]))
        if not summary:
            section_name = str(section.get("name") or section.get("slug") or "").strip()
            if section_name.startswith("补充归类 "):
                summary = "当前用于承接尚未稳定并入主模块的文件与目录。"
            elif section_name:
                summary = f"主要负责 {section_name} 相关能力。"
        if summary:
            summary_line = f"职责：{_truncate_sentence(summary, max_chars=160)}"
            replaced = False
            new_body: list[str] = []
            for line in cloned["body_lines"]:
                if not replaced and re.match(r"^职责[:：]\s*", line.strip()):
                    new_body.append(summary_line)
                    replaced = True
                else:
                    new_body.append(line)
            if not replaced:
                insert_at = 1 if new_body and re.match(r"^文件[:：]\s*", new_body[0].strip()) else 0
                new_body.insert(insert_at, summary_line)
            cloned["body_lines"] = new_body
        injected.append(cloned)
    return injected


def _enrich_tree_summaries_with_cognition(
    tree: NavigationTree,
    *,
    sections_by_slug: dict[str, dict[str, Any]],
    cognition_facts: ModuleCognitionFacts,
) -> NavigationTree:
    facts_by_slug = {fact.slug: fact for fact in cognition_facts.modules}
    enriched_areas: list[AreaRecord] = []
    derived_area_summaries: list[str] = []

    for area in tree.areas:
        summary = (area.summary or "").strip()
        if _is_generic_area_summary(summary):
            module_names: list[str] = []
            module_summaries: list[str] = []
            for slug in area.module_slugs:
                section = sections_by_slug.get(slug, {})
                module_names.append(str(section.get("name") or slug))
                fact = facts_by_slug.get(slug)
                module_summary = _module_nav_summary_from_cognition_fact(fact)
                if module_summary:
                    module_summaries.append(_truncate_sentence(module_summary, max_chars=72))
                if len(module_summaries) >= 2:
                    break
            if module_summaries:
                prefix = (
                    f"围绕{'、'.join(module_names[:2])}等模块，"
                    if len(area.module_slugs) > 1
                    else ""
                )
                summary = prefix + "；".join(module_summaries[:2])
            elif module_names:
                summary = f"围绕{'、'.join(module_names[:3])}等模块展开。"
        summary = _truncate_sentence(summary, max_chars=140)
        if summary and not area.is_fallback:
            derived_area_summaries.append(summary)
        enriched_areas.append(area.model_copy(update={"summary": summary}))

    project_summary = (tree.project_summary or "").strip()
    if _is_generic_project_summary(project_summary):
        non_fallback_names = [area.display_name for area in enriched_areas if not area.is_fallback]
        project_parts: list[str] = []
        if non_fallback_names:
            project_parts.append(
                f"项目围绕{'、'.join(non_fallback_names[:3])}等区域组织代码与能力。"
            )
        if derived_area_summaries:
            project_parts.append(_truncate_sentence("；".join(derived_area_summaries[:2]), max_chars=180))
        project_summary = "\n".join(part for part in project_parts if part).strip()
        if not project_summary:
            project_summary = tree.project_summary

    return tree.model_copy(update={
        "project_summary": project_summary,
        "areas": enriched_areas,
    })


def _append_markdown_list_section(
    lines: list[str],
    heading: str,
    items: list[str],
    *,
    omit_if_empty: bool = False,
) -> None:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if omit_if_empty and not normalized:
        return
    lines.extend(["", f"## {heading}"])
    for item in normalized:
        lines.append(f"- {item}")


def _render_module_md_from_cognition(
    module_name: str,
    module_entries: list[IndexEntry],
    fact: Any | None,
) -> str:
    title = str(getattr(fact, "display_name", None) or module_name).strip() or module_name
    coverage = [str(entry.file_meta.path) for entry in module_entries]
    lines = [f"# {title}", "", "## 覆盖文件"]
    for path in coverage:
        lines.append(f"- {path}")

    _append_markdown_list_section(
        lines,
        "核心职责",
        list(getattr(fact, "core_responsibility", []) or []),
    )
    _append_markdown_list_section(
        lines,
        "关键流程",
        list(getattr(fact, "key_flow", []) or []),
    )
    _append_markdown_list_section(
        lines,
        "外部协作",
        list(getattr(fact, "external_collaboration", []) or []),
    )
    _append_markdown_list_section(
        lines,
        "风险与约束",
        list(getattr(fact, "risks_constraints", []) or []),
    )
    _append_markdown_list_section(
        lines,
        "关键符号",
        list(getattr(fact, "key_anchors", []) or []),
        omit_if_empty=True,
    )
    return "\n".join(lines).rstrip() + "\n"


async def _write_module_files_from_cognition(
    module_specs: list[tuple[str, str, list[IndexEntry]]],
    modules_dir: Path,
    cognition_facts: ModuleCognitionFacts,
) -> None:
    facts_by_slug = {item.slug: item for item in cognition_facts.modules}
    for module_name, slug, module_entries in module_specs:
        fact = facts_by_slug.get(slug)
        if fact is None:
            content = _build_fallback_module_md(module_name, module_entries)
        else:
            content = _render_module_md_from_cognition(
                module_name, module_entries, fact
            )
        await _atomic_write_text(modules_dir / f"{slug}.md", content.rstrip() + "\n")


async def _write_empty_annotations(root_path: Path) -> None:
    index_path = _annotation_index_path(root_path)
    areas_dir = _annotation_areas_dir(root_path)
    await _atomic_write_text(index_path, "# 项目认知导航\n\n当前暂无可导航模块。\n")
    await _cleanup_stale_area_docs(areas_dir, set())
    tree_path = _navigation_tree_path(root_path)
    if tree_path.exists():
        await asyncio.to_thread(tree_path.unlink, missing_ok=True)
    await _cleanup_stale_annotation_docs(root_path, {Path(ANNOTATIONS_INDEX_FILE)})


async def _prepare_annotation_payload(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    *,
    model: str | None,
) -> tuple[
    dict[str, IndexEntry],
    list[dict[str, Any]],
    list[tuple[str, str, list[IndexEntry]]],
]:
    """执行模块发现与 registry 稳定化，返回后续渲染所需 payload。"""
    index_content = await _generate_index_md(entries, project_meta, model=model)
    if not index_content:
        logger.warning("认知导航 index.md 不可用，使用索引级回退模板")
        index_content = _build_fallback_index_md(entries)

    entries_map = {str(entry.file_meta.path): entry for entry in entries}
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
        _, sections = _split_index_md(index_content)
        valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    return entries_map, valid_sections, module_specs


async def _persist_annotation_outputs(
    *,
    root_path: Path,
    tree: NavigationTree,
    sections_by_slug: dict[str, dict[str, Any]],
    module_specs: list[tuple[str, str, list[IndexEntry]]],
) -> None:
    index_path = _annotation_index_path(root_path)
    areas_dir = _annotation_areas_dir(root_path)
    await _atomic_write_text(
        index_path,
        _render_hierarchical_index_md(tree, sections_by_slug),
    )
    await _atomic_write_text(
        _navigation_tree_path(root_path),
        json.dumps(tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )
    areas_dir.mkdir(parents=True, exist_ok=True)
    for area in tree.areas:
        await _atomic_write_text(
            areas_dir / f"{area.slug}.md",
            _render_area_md(area, sections_by_slug),
        )

    await _persist_navigation_area_cache(root_path, tree)

    valid_area_slugs = {area.slug for area in tree.areas}
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
        Path(ANNOTATIONS_AREAS_DIR) / f"{area.slug}.md" for area in tree.areas
    )
    await _cleanup_stale_annotation_docs(root_path, expected_files)


def _build_sections_from_navigation_tree(
    tree: NavigationTree,
    entries_map: dict[str, IndexEntry],
) -> list[dict[str, Any]]:
    """将 navigation_tree 中的模块边界与默认从属恢复为 section 列表。"""
    sections: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    assignment = _assign_navigation_tree_paths(tree, entries_map)

    for area in tree.areas:
        for module in area.modules:
            file_paths = list(assignment["module_paths"].get(module.slug, []))
            if not file_paths:
                continue
            if module.slug in seen_slugs:
                raise ValueError(f"navigation_tree 模块 slug 重复: {module.slug}")

            body_lines = [f"文件：{', '.join(file_paths)}"]
            summary = _truncate_sentence(str(module.summary or '').strip(), max_chars=160)
            if summary:
                body_lines.append(f"职责：{summary}")
            body_lines.append(f"详细认知：.pce/annotations/modules/{module.slug}.md")
            sections.append({
                "name": module.display_name,
                "slug": module.slug,
                "file_paths": file_paths,
                "body_lines": body_lines,
            })
            seen_slugs.add(module.slug)

    if assignment["duplicate_paths"]:
        detail = assignment["duplicate_details"][0]
        raise ValueError(
            "navigation_tree 文件重复挂载: "
            f"{detail['path']} 同时属于 {' / '.join(detail['modules'])}"
        )
    if assignment["invalid_paths"]:
        raise ValueError(
            "navigation_tree 含非法 path_rules: " + ", ".join(assignment["invalid_paths"][:8])
        )

    return sections


def _build_section_slug_aliases(
    source_sections: list[dict[str, Any]],
    target_sections: list[dict[str, Any]],
) -> dict[str, str]:
    """根据前后两轮 section 的覆盖文件建立 slug 别名映射。"""
    target_by_files: dict[tuple[str, ...], str] = {}
    for section in target_sections:
        file_key = tuple(sorted(_dedupe_keep_order(section.get("file_paths", []))))
        if file_key:
            target_by_files[file_key] = section["slug"]

    aliases: dict[str, str] = {}
    for section in source_sections:
        file_key = tuple(sorted(_dedupe_keep_order(section.get("file_paths", []))))
        target_slug = target_by_files.get(file_key)
        if file_key and target_slug and target_slug != section["slug"]:
            aliases[section["slug"]] = target_slug
    return aliases


def _sync_tree_modules_from_sections(
    tree: NavigationTree,
    sections_by_slug: dict[str, dict[str, Any]],
) -> NavigationTree:
    """按 sections_by_slug 回填 tree 中的模块入口信息。"""
    enriched_areas: list[AreaRecord] = []
    for area in tree.areas:
        modules = []
        for slug in area.module_slugs:
            original = _module_nav_record_by_slug(area, slug)
            if original is None:
                continue
            modules.append(ModuleNavRecord(
                module_type=original.module_type,
                slug=slug,
                display_name=original.display_name,
                summary=_extract_section_summary(sections_by_slug.get(slug, {})) or str(original.summary or "").strip(),
                include=list(original.include),
                exclude=list(original.exclude),
            ))
        enriched_areas.append(area.model_copy(update={"modules": modules}))
    return tree.model_copy(update={"areas": enriched_areas})


def _normalize_navigation_tree_module_slugs(tree: NavigationTree) -> NavigationTree:
    """规范化 tree 内 module slug，确保全局唯一且稳定。"""
    used_slugs: set[str] = set()
    remapped_areas: list[AreaRecord] = []

    for area in tree.areas:
        alias_map: dict[str, str] = {}
        remapped_modules: list[ModuleNavRecord] = []
        for module in area.modules:
            base_slug = str(module.slug or "").strip()
            candidate_slug = base_slug or ModuleRegistryManager.normalize_slug(
                str(module.display_name or "").strip()
            )
            unique_slug = _ensure_unique_section_slug(candidate_slug or "module", used_slugs)
            used_slugs.add(unique_slug)
            alias_map[base_slug] = unique_slug
            remapped_modules.append(module.model_copy(update={"slug": unique_slug}))

        module_slugs: list[str] = []
        for module in remapped_modules:
            if module.slug not in module_slugs:
                module_slugs.append(module.slug)

        recommended_order: list[str] = []
        for slug in area.recommended_order:
            mapped = alias_map.get(slug, slug)
            if mapped in module_slugs and mapped not in recommended_order:
                recommended_order.append(mapped)
        for slug in module_slugs:
            if slug not in recommended_order:
                recommended_order.append(slug)

        remapped_areas.append(area.model_copy(update={
            "modules": remapped_modules,
            "module_slugs": module_slugs,
            "recommended_order": recommended_order,
        }))

    return tree.model_copy(update={"areas": remapped_areas})


def _remap_navigation_tree_model_slugs(
    tree: NavigationTree,
    slug_aliases: dict[str, str],
    sections_by_slug: dict[str, dict[str, Any]],
) -> NavigationTree:
    """将 tree 中的 module slug 统一映射为稳定 slug。"""
    remapped_areas: list[AreaRecord] = []
    for area in tree.areas:
        remapped_modules: list[ModuleNavRecord] = []
        seen_module_slugs: set[str] = set()
        for module in area.modules:
            mapped_slug = slug_aliases.get(module.slug, module.slug)
            cloned = module.model_copy(update={"slug": mapped_slug})
            if cloned.slug in seen_module_slugs:
                continue
            remapped_modules.append(cloned)
            seen_module_slugs.add(cloned.slug)
        module_slugs = _dedupe_keep_order([
            slug_aliases.get(slug, slug) for slug in area.module_slugs
        ])
        recommended_order = _dedupe_keep_order([
            slug_aliases.get(slug, slug) for slug in area.recommended_order
        ])
        remapped_areas.append(area.model_copy(update={
            "modules": remapped_modules,
            "module_slugs": module_slugs,
            "recommended_order": recommended_order,
        }))
    remapped = tree.model_copy(update={"areas": remapped_areas})
    return _sync_tree_modules_from_sections(remapped, sections_by_slug)


def _remap_module_cognition_facts_model(
    cognition_facts: ModuleCognitionFacts,
    slug_aliases: dict[str, str],
    sections_by_slug: dict[str, dict[str, Any]],
) -> ModuleCognitionFacts:
    """将模块认知事实统一映射为稳定 slug。"""
    payload = _remap_module_cognition_payload_slugs(
        cognition_facts.model_dump(mode="json"),
        slug_aliases,
    )
    normalized = _coerce_module_cognition_facts(
        payload,
        expected_slugs=set(sections_by_slug),
    )
    modules = [
        fact.model_copy(update={
            "display_name": str(
                sections_by_slug.get(fact.slug, {}).get("name") or fact.display_name
            )
        })
        for fact in normalized.modules
    ]
    return ModuleCognitionFacts(modules=modules)


def _coerce_topology_navigation_tree(
    payload: dict[str, Any],
    *,
    facts: dict[str, Any],
    active_slugs: set[str] | None,
) -> NavigationTree:
    def _listish_strings(value: Any) -> list[str]:
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                text = str(item).strip()
                if not text:
                    continue
                parts = [part.strip() for part in re.split(r"[\n,]+", text) if part.strip()]
                items.extend(parts)
            return _dedupe_keep_order(items)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            return _dedupe_keep_order([
                part.strip() for part in re.split(r"[\n,]+", text) if part.strip()
            ])
        return []

    def _normalize_modules(value: Any) -> list[dict[str, Any]]:
        normalized_modules: list[dict[str, Any]] = []
        if not isinstance(value, list):
            return normalized_modules
        for raw_module in value:
            if not isinstance(raw_module, dict):
                continue
            display_name = " ".join(str(raw_module.get("display_name") or "").split())
            module_type = _normalize_module_nav_type(raw_module.get("module_type"))
            include = _listish_strings(raw_module.get("include"))
            exclude = _listish_strings(raw_module.get("exclude"))
            if not display_name:
                continue
            if module_type != "residual" and not include:
                continue
            slug = " ".join(str(raw_module.get("slug") or "").split())
            if not slug:
                slug = ModuleRegistryManager.normalize_slug(display_name)
            summary = " ".join(str(raw_module.get("summary") or "").split())
            normalized_modules.append({
                "slug": slug,
                "display_name": display_name,
                "summary": _truncate_sentence(summary, max_chars=160) if summary else "",
                "module_type": module_type,
                "include": include,
                "exclude": exclude,
            })
        return _merge_navigation_module_payloads(normalized_modules)

    tree_payload = dict(payload)
    raw_areas = payload.get("areas")
    normalized_areas: list[dict[str, Any]] = []
    if isinstance(raw_areas, list):
        for raw_area in raw_areas:
            if not isinstance(raw_area, dict):
                continue
            area = dict(raw_area)
            modules = _normalize_modules(area.get("modules"))
            area["modules"] = modules
            area["module_slugs"] = (
                [module["slug"] for module in modules]
                if modules
                else _listish_strings(area.get("module_slugs"))
            )
            recommended_order = area.get("recommended_order")
            area["recommended_order"] = (
                _listish_strings(recommended_order)
                if isinstance(recommended_order, (list, str))
                else []
            )
            if not area["recommended_order"] and area["module_slugs"]:
                area["recommended_order"] = list(area["module_slugs"])
            area["source_prefixes"] = _listish_strings(area.get("source_prefixes"))
            normalized_areas.append(area)
    tree_payload["areas"] = normalized_areas
    tree_payload["source_digest"] = str(facts["source_digest"])
    tree_payload["generated_at"] = datetime.now(UTC).isoformat()
    tree = NavigationTree.model_validate(tree_payload)
    effective_active_slugs = set(active_slugs or {
        slug for area in tree.areas for slug in area.module_slugs
    })
    tree = _repair_navigation_tree(tree, effective_active_slugs)
    errors = _validate_navigation_tree(tree, effective_active_slugs)
    if errors:
        raise ValueError(" | ".join(errors))
    return tree


def _remap_module_cognition_payload_slugs(
    payload: dict[str, Any],
    slug_aliases: dict[str, str],
) -> dict[str, Any]:
    """将 module_cognition_facts payload 中的旧 module slug 归一化到稳定 slug。"""
    if not slug_aliases:
        return payload
    normalized = dict(payload)
    raw_modules = payload.get("modules")
    remapped_modules: list[dict[str, Any]] = []
    if isinstance(raw_modules, list):
        for raw_item in raw_modules:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            slug = str(item.get("slug") or "").strip()
            if slug:
                item["slug"] = slug_aliases.get(slug, slug)
            remapped_modules.append(item)
    normalized["modules"] = remapped_modules
    return normalized


def _coerce_module_cognition_facts(
    payload: dict[str, Any],
    *,
    expected_slugs: set[str],
) -> ModuleCognitionFacts:
    def _listish(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        return []

    def _normalize_lines(
        items: list[str],
        *,
        limit: int,
        max_chars: int | None = None,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in items:
            text = " ".join(str(raw).strip().split())
            if not text:
                continue
            if max_chars is not None and len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."
            if text in seen:
                continue
            seen.add(text)
            normalized.append(text)
            if len(normalized) >= limit:
                break
        return normalized

    normalized_payload = dict(payload)
    normalized_modules: list[dict[str, Any]] = []
    for raw_item in payload.get("modules", []) if isinstance(payload.get("modules"), list) else []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        for key in (
            "core_responsibility",
            "key_flow",
            "external_collaboration",
            "risks_constraints",
            "key_anchors",
        ):
            item[key] = _listish(item.get(key))
        normalized_modules.append(item)
    normalized_payload["modules"] = normalized_modules

    facts = ModuleCognitionFacts.model_validate(normalized_payload)
    filtered = []
    seen: set[str] = set()
    for item in facts.modules:
        if item.slug not in expected_slugs or item.slug in seen:
            continue
        seen.add(item.slug)
        filtered.append(
            item.model_copy(
                update={
                    "core_responsibility": _normalize_lines(
                        list(item.core_responsibility), limit=3, max_chars=180
                    ),
                    "key_flow": _normalize_lines(
                        list(item.key_flow), limit=3, max_chars=220
                    ),
                    "external_collaboration": _normalize_lines(
                        list(item.external_collaboration), limit=3, max_chars=180
                    ),
                    "risks_constraints": _normalize_lines(
                        list(item.risks_constraints), limit=3, max_chars=180
                    ),
                    "key_anchors": _normalize_lines(
                        list(item.key_anchors), limit=2, max_chars=120
                    ),
                }
            )
        )
    return ModuleCognitionFacts(modules=filtered)


def _is_topology_default_core_area(area: AreaRecord) -> bool:
    return (
        not area.is_fallback
        and area.display_name == "核心模块"
        and area.summary == "模型未给出稳定区域划分时的默认主区域。"
    )


def _assert_navigation_tree_not_fallback_only(tree: NavigationTree) -> None:
    non_fallback_areas = [area for area in tree.areas if not area.is_fallback]
    if len(non_fallback_areas) == 1 and _is_topology_default_core_area(non_fallback_areas[0]):
        raise ValueError("navigation_tree 退化为默认 core/fallback 结构，需重新生成")


def _collect_area_module_doc_status(
    area: AreaRecord,
    *,
    modules_dir: Path,
    module_specs_by_slug: dict[str, tuple[str, str, list[IndexEntry]]],
    area_started_ns: int,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    completed: list[str] = []
    pending: list[str] = []
    invalid: dict[str, list[str]] = {}

    for slug in area.module_slugs:
        path = modules_dir / f"{slug}.md"
        if not path.exists():
            pending.append(slug)
            continue
        try:
            stat = path.stat()
        except OSError:
            pending.append(slug)
            continue
        if stat.st_mtime_ns < area_started_ns:
            pending.append(slug)
            continue

        _, _, module_entries = module_specs_by_slug[slug]
        expected_paths = [str(entry.file_meta.path) for entry in module_entries]
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            invalid[slug] = [f"读取失败: {exc}"]
            continue

        errors = validate_module_annotation_markdown(
            content,
            expected_file_paths=expected_paths,
            require_core_responsibility=True,
        )
        if errors:
            invalid[slug] = errors
            continue
        completed.append(slug)

    return completed, pending, invalid


async def _write_area_fallback_modules(
    area: AreaRecord,
    *,
    modules_dir: Path,
    module_specs_by_slug: dict[str, tuple[str, str, list[IndexEntry]]],
    target_slugs: list[str],
) -> None:
    for slug in target_slugs:
        module_name, _, module_entries = module_specs_by_slug[slug]
        await _atomic_write_text(
            modules_dir / f"{slug}.md",
            _build_fallback_module_md(module_name, module_entries),
        )


async def _write_empty_module_skeletons(
    module_specs: list[tuple[str, str, list[IndexEntry]]],
    modules_dir: Path,
) -> None:
    for module_name, slug, module_entries in module_specs:
        await _atomic_write_text(
            modules_dir / f"{slug}.md",
            _build_empty_module_skeleton_md(module_name, module_entries),
        )


async def _write_annotations_via_topology_agent(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    *,
    serena_client: SerenaClient,
    model: str | None,
) -> None:
    """使用 topology cognition agent 生成轻量 init 认知。"""
    if not entries:
        await _write_empty_annotations(root_path)
        return

    modules_dir = _annotation_modules_dir(root_path)
    stable_tree, stable_sections_by_slug, module_specs = await _build_topology_navigation_bundle(
        entries=entries,
        root_path=root_path,
        serena_client=serena_client,
        model=model,
    )

    await _persist_annotation_outputs(
        root_path=root_path,
        tree=stable_tree,
        sections_by_slug=stable_sections_by_slug,
        module_specs=module_specs,
    )
    await _write_empty_module_skeletons(module_specs, modules_dir)
    logger.info(
        "Topology agent 认知导航写入完成: %d 个区域, %d 个模块骨架文档",
        len(stable_tree.areas),
        len(module_specs),
    )


async def _build_topology_navigation_bundle(
    *,
    entries: list[IndexEntry],
    root_path: Path,
    serena_client: SerenaClient,
    model: str | None,
) -> tuple[NavigationTree, dict[str, dict[str, Any]], list[tuple[str, str, list[IndexEntry]]]]:
    entries_map = {str(entry.file_meta.path): entry for entry in entries}
    discovery_facts = _build_topology_discovery_facts(entries_map, root_path)

    agent = TopologyCognitionAgent(
        project_root=root_path,
        discovery_facts=discovery_facts,
        model=model,
    )
    messages = agent.build_initial_messages()
    raw_tree: NavigationTree | None = None
    valid_sections: list[dict[str, Any]] | None = None
    module_specs: list[tuple[str, str, list[IndexEntry]]] | None = None
    stable_tree: NavigationTree | None = None
    stable_sections_by_slug: dict[str, dict[str, Any]] | None = None
    repair_attempts = 0
    max_repair_rounds = 6

    last_navigation_exc: BaseException | None = None
    logger.info("topology init stage start: navigation_tree")
    try:
        navigation_payload = await agent.run_stage(
            stage="navigation_tree",
            messages=messages,
            serena_client=serena_client,
        )
        if not isinstance(navigation_payload, dict):
            raise ValueError("navigation_tree payload 必须为 object")
        await _persist_topology_stage_debug(
            root_path,
            stage="navigation_tree",
            payload=navigation_payload,
            extra={"attempt": 1},
        )

        raw_tree = _coerce_topology_navigation_tree(
            navigation_payload,
            facts={"source_digest": "topology-init"},
            active_slugs=None,
        )
        raw_tree = _normalize_navigation_tree_module_slugs(raw_tree)
        _assert_navigation_tree_not_fallback_only(raw_tree)
        last_metrics: tuple[int, int, int] | None = None
        while True:
            payload_issues = _collect_navigation_payload_issues(raw_tree, entries_map)
            raw_sections = _build_sections_from_navigation_tree(raw_tree, entries_map)
            valid_sections, raw_module_specs = _prepare_module_specs(raw_sections, entries_map)
            if not valid_sections:
                raise ValueError("navigation_tree 解析后未得到有效模块规格")

            raw_sections_by_slug = {section["slug"]: section for section in valid_sections}
            raw_active_slugs = {slug for _, slug, _ in raw_module_specs}
            raw_tree = _repair_navigation_tree(
                raw_tree.model_copy(update={
                    "generated_at": datetime.now(UTC),
                    "source_digest": _compute_source_digest(valid_sections),
                }),
                raw_active_slugs,
            )
            raw_tree = _sync_tree_modules_from_sections(raw_tree, raw_sections_by_slug)
            _assert_navigation_tree_not_fallback_only(raw_tree)
            tree_errors = _validate_navigation_tree(raw_tree, raw_active_slugs)
            if tree_errors:
                raise ValueError(" | ".join(tree_errors))

            coverage_issues = _collect_navigation_tree_coverage_issues(
                valid_sections,
                entries_map,
            )
            metrics = (
                len(coverage_issues["unassigned_paths"]),
                len(payload_issues["duplicate_paths"]),
                len(payload_issues["invalid_paths"]),
            )
            if metrics == (0, 0, 0):
                break

            repair_attempts += 1
            if repair_attempts > max_repair_rounds:
                issue_parts: list[str] = []
                if coverage_issues["unassigned_paths"]:
                    issue_parts.append(f"仍有 {len(coverage_issues['unassigned_paths'])} 个文件未挂载")
                if payload_issues["duplicate_paths"]:
                    issue_parts.append(f"仍有 {len(payload_issues['duplicate_paths'])} 个文件重复挂载")
                if payload_issues["invalid_paths"]:
                    issue_parts.append(f"仍有 {len(payload_issues['invalid_paths'])} 个非法路径")
                raise ValueError(" | ".join(issue_parts) or "navigation_tree 覆盖修复失败")

            if last_metrics is not None and not (
                metrics[0] < last_metrics[0]
                or metrics[1] < last_metrics[1]
                or metrics[2] < last_metrics[2]
            ):
                raise ValueError(
                    "navigation_repair 未继续收敛: "
                    f"unassigned={metrics[0]} duplicate={metrics[1]} invalid={metrics[2]}"
                )

            last_metrics = metrics
            missing_paths_tree = _build_topology_missing_paths_tree(
                coverage_issues["unassigned_paths"],
                entries_map,
            )
            repair_payload = await agent.run_stage(
                stage="navigation_repair",
                messages=messages,
                serena_client=serena_client,
                stage_context={
                    "current_navigation_tree": raw_tree.model_dump(mode="json"),
                    "missing_paths_tree": missing_paths_tree,
                    "validation_feedback": {
                        **coverage_issues,
                        **payload_issues,
                    },
                },
                user_prompt=_build_navigation_repair_feedback(
                    attempt=repair_attempts,
                    max_attempts=max_repair_rounds,
                    issues={
                        **coverage_issues,
                        **payload_issues,
                    },
                ),
            )
            if not isinstance(repair_payload, dict):
                raise ValueError("navigation_repair payload 必须为 object")
            await _persist_topology_stage_debug(
                root_path,
                stage="navigation_repair",
                payload=repair_payload,
                extra={"attempt": 1, "repair_attempt": repair_attempts, "metrics": metrics},
            )
            raw_tree = _coerce_topology_navigation_tree(
                repair_payload,
                facts={"source_digest": "topology-init-repair"},
                active_slugs=None,
            )
            raw_tree = _normalize_navigation_tree_module_slugs(raw_tree)
            _assert_navigation_tree_not_fallback_only(raw_tree)

        stable_sections_by_slug = {
            section["slug"]: section for section in valid_sections
        }
        module_specs = raw_module_specs
        stable_active_slugs = {slug for _, slug, _ in module_specs}
        stable_tree = _repair_navigation_tree(
            raw_tree.model_copy(update={
                "generated_at": datetime.now(UTC),
                "source_digest": _compute_source_digest(valid_sections),
            }),
            stable_active_slugs,
        )
        stable_tree = _sync_tree_modules_from_sections(stable_tree, stable_sections_by_slug)
        _assert_navigation_tree_not_fallback_only(stable_tree)
        tree_errors = _validate_navigation_tree(stable_tree, stable_active_slugs)
        if tree_errors:
            raise ValueError(" | ".join(tree_errors))

        logger.info("topology init stage ok: navigation_tree")
    except Exception as exc:
        last_navigation_exc = exc
        await _persist_topology_debug_error(
            root_path,
            stage="navigation_tree",
            message=_format_exception_brief(exc),
            extra={"attempt": 1},
        )
        raise RuntimeError(f"navigation_tree stage failed: {exc}") from exc

    if (
        raw_tree is None
        or valid_sections is None
        or module_specs is None
        or stable_tree is None
        or stable_sections_by_slug is None
    ):
        raise RuntimeError(f"navigation_tree stage failed: {last_navigation_exc}")

    return stable_tree, stable_sections_by_slug, module_specs


async def _write_annotations_legacy(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    model: str | None = None,
) -> None:
    """全量生成并写入三层树状认知导航。"""
    modules_dir = _annotation_modules_dir(root_path)

    if not entries:
        await _write_empty_annotations(root_path)
        return

    entries_map, valid_sections, module_specs = await _prepare_annotation_payload(
        entries,
        project_meta,
        root_path,
        model=model,
    )

    sections_by_slug = {section["slug"]: section for section in valid_sections}
    active_slugs = {slug for _, slug, _ in module_specs}
    facts = _build_navigation_facts(valid_sections, entries_map, root_path)

    tree = await _generate_navigation_tree(facts, active_slugs, model=model)
    if tree is None:
        logger.warning("导航树生成失败，回退到确定性 fallback 方案")
        tree = _build_fallback_navigation_tree(valid_sections, active_slugs)

    tree = _repair_navigation_tree(
        tree.model_copy(update={"source_digest": str(facts["source_digest"])}),
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

    injected_sections = _inject_section_summaries_from_cognition(
        valid_sections,
        ModuleCognitionFacts(),
    )
    injected_sections_by_slug = {
        section["slug"]: section for section in injected_sections
    }
    enriched_tree = _enrich_tree_summaries_with_cognition(
        tree,
        sections_by_slug=injected_sections_by_slug,
        cognition_facts=ModuleCognitionFacts(),
    )

    await _write_module_files(module_specs, modules_dir, model)
    await _persist_annotation_outputs(
        root_path=root_path,
        tree=enriched_tree,
        sections_by_slug=injected_sections_by_slug,
        module_specs=module_specs,
    )

    logger.info(
        "树状认知导航写入完成: %d 个区域, %d 个模块文档",
        len(enriched_tree.areas),
        len(module_specs),
    )


async def _write_annotations(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    model: str | None = None,
    *,
    serena_client: SerenaClient | None = None,
) -> None:
    """全量生成认知导航；有 Serena 时强制走 topology agent。"""
    if serena_client is not None:
        await _write_annotations_via_topology_agent(
            entries,
            project_meta,
            root_path,
            serena_client=serena_client,
            model=model,
        )
        return

    await _write_annotations_legacy(
        entries,
        project_meta,
        root_path,
        model=model,
    )


async def refresh_navigation_from_snapshot(
    *,
    root_path: Path,
    serena_client: SerenaClient,
    model: str | None = None,
) -> dict[str, Any]:
    """基于当前 index snapshot 重建 navigation_tree，并重渲染 index/areas。

    不触碰现有 modules 文档正文。
    """
    snapshot = await load_index(root_path=root_path)
    if snapshot is None:
        raise RuntimeError("当前无可用 index snapshot，无法刷新 navigation")
    entries = list(snapshot.entries)
    if not entries:
        await _write_empty_annotations(root_path)
        return {"areas": 0, "modules": 0}

    stable_tree, stable_sections_by_slug, module_specs = await _build_topology_navigation_bundle(
        entries=entries,
        root_path=root_path,
        serena_client=serena_client,
        model=model,
    )
    await _persist_annotation_outputs(
        root_path=root_path,
        tree=stable_tree,
        sections_by_slug=stable_sections_by_slug,
        module_specs=module_specs,
    )
    return {"areas": len(stable_tree.areas), "modules": len(module_specs)}


def _path_exists_in_baseline(root_path: Path, rel_path: str) -> bool:
    baseline_path = root_path / ".pce" / "baselines" / "files" / f"{rel_path}.json"
    return baseline_path.exists()


def _matches_module_rules(path: str, module: ModuleNavRecord) -> bool:
    resolved_paths, _ = _expand_module_path_rules(module, {path})
    if resolved_paths:
        return True
    if module.module_type == "directory":
        return any(_path_matches_prefix(path, root) for root in _directory_roots_for_module(module))
    return False


def _assign_deleted_path_to_tree(
    tree: NavigationTree,
    rel_path: str,
) -> str | None:
    matched_modules: list[ModuleNavRecord] = []
    fallback_residual: ModuleNavRecord | None = None
    for area in tree.areas:
        residual_module = next((module for module in area.modules if module.module_type == "residual"), None)
        for module in area.modules:
            if module.module_type == "residual":
                continue
            if _matches_module_rules(rel_path, module):
                matched_modules.append(module)
        if area.is_fallback and residual_module is not None:
            fallback_residual = residual_module

    unique = _dedupe_keep_order([module.slug for module in matched_modules])
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        ranked = sorted(
            matched_modules,
            key=lambda module: _path_specificity(rel_path, module),
            reverse=True,
        )
        if len(ranked) == 1 or _path_specificity(rel_path, ranked[0]) > _path_specificity(rel_path, ranked[1]):
            return ranked[0].slug
        return None

    for area in tree.areas:
        if area.is_fallback:
            continue
        if not any(_path_matches_prefix(rel_path, prefix) for prefix in area.source_prefixes):
            continue
        residual_module = next((module for module in area.modules if module.module_type == "residual"), None)
        if residual_module is not None:
            return residual_module.slug

    return fallback_residual.slug if fallback_residual is not None else None


def _collect_empty_module_slugs(
    tree: NavigationTree,
    assignment: dict[str, Any],
) -> list[str]:
    module_paths = assignment.get("module_paths") or {}
    empty_slugs: list[str] = []
    for area in tree.areas:
        for module in area.modules:
            if module_paths.get(module.slug):
                continue
            empty_slugs.append(module.slug)
    return sorted(_dedupe_keep_order(empty_slugs))


def _build_incremental_navigation_facts(
    *,
    root_path: Path,
    tree: NavigationTree,
    entries_map: dict[str, IndexEntry],
    changed_files: list[str],
    deleted_files: list[str],
) -> dict[str, Any]:
    assignment = _assign_navigation_tree_paths(tree, entries_map)
    created_files = [
        path for path in changed_files
        if path in entries_map and not _path_exists_in_baseline(root_path, path)
    ]
    modified_files = [path for path in changed_files if path not in created_files]

    created_covered = sorted(path for path in created_files if assignment["path_owner"].get(path))
    created_uncovered = sorted(path for path in created_files if not assignment["path_owner"].get(path))

    deleted_assignments: list[dict[str, Any]] = []
    for path in deleted_files:
        deleted_assignments.append({
            "path": path,
            "owner_slug": _assign_deleted_path_to_tree(tree, path),
        })

    impacted_module_slugs = sorted(_dedupe_keep_order([
        *[str(assignment["path_owner"].get(path) or "").strip() for path in created_covered],
        *[str(item.get("owner_slug") or "").strip() for item in deleted_assignments],
    ]))
    impacted_module_slugs = [slug for slug in impacted_module_slugs if slug]
    impacted_area_slugs = sorted(_dedupe_keep_order([
        area.slug
        for area in tree.areas
        if any(module.slug in impacted_module_slugs for module in area.modules)
    ]))

    structural_paths = _dedupe_keep_order([*created_files, *deleted_files])
    structural_tree = _build_topology_missing_paths_tree(
        [path for path in structural_paths if path in entries_map],
        entries_map,
    )

    return {
        "current_navigation_tree": tree.model_dump(mode="json"),
        "structural_changes": {
            "created_files": created_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "created_covered": created_covered,
            "created_uncovered": created_uncovered,
            "deleted_assignments": deleted_assignments,
            "impacted_module_slugs": impacted_module_slugs,
            "impacted_area_slugs": impacted_area_slugs,
            "structural_tree": structural_tree,
        },
        "validation_feedback": {
            "unassigned_paths": list(assignment["unassigned_paths"]),
            "duplicate_paths": list(assignment["duplicate_paths"]),
            "duplicate_details": list(assignment["duplicate_details"]),
            "invalid_paths": list(assignment["invalid_paths"]),
            "area_residuals": list(assignment["area_residuals"]),
            "too_many_file_centered_modules": list(assignment["too_many_file_centered_modules"]),
            "empty_module_slugs": _collect_empty_module_slugs(tree, assignment),
        },
    }


def _coerce_incremental_navigation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    decision = str(payload.get("decision") or "").strip()
    if decision not in {"no_change", "module_update", "area_rebuild", "full_rebuild"}:
        raise ValueError("decision 必须是 no_change/module_update/area_rebuild/full_rebuild")
    rationale = " ".join(str(payload.get("rationale") or "").split())
    navigation_tree = payload.get("navigation_tree")
    return {
        "decision": decision,
        "rationale": rationale,
        "navigation_tree": navigation_tree,
    }


async def _run_incremental_navigation_update(
    *,
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    tree: NavigationTree,
    changed_files: list[str],
    deleted_files: list[str],
    model: str | None,
    serena_client: SerenaClient,
) -> dict[str, Any]:
    del project_meta
    entries_map = {str(entry.file_meta.path): entry for entry in entries}
    facts = _build_incremental_navigation_facts(
        root_path=root_path,
        tree=tree,
        entries_map=entries_map,
        changed_files=changed_files,
        deleted_files=deleted_files,
    )

    if not facts["structural_changes"]["created_files"] and not facts["structural_changes"]["deleted_files"]:
        return {"decision": "no_change", "rationale": "only_content_changes"}
    if (
        facts["structural_changes"]["created_files"]
        and not facts["structural_changes"]["created_uncovered"]
        and not deleted_files
    ):
        return {"decision": "no_change", "rationale": "created_files_already_covered"}

    agent = TopologyCognitionAgent(
        project_root=root_path,
        discovery_facts=facts,
        model=model,
    )
    messages = agent.build_initial_messages()
    payload: dict[str, Any] | None = None
    last_exc: Exception | None = None
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        stage_name = "navigation_incremental" if attempt == 1 else "navigation_incremental_repair"
        try:
            raw_payload = await agent.run_stage(
                stage=stage_name,
                messages=messages,
                serena_client=serena_client,
                stage_context=facts,
            )
            payload = _coerce_incremental_navigation_payload(raw_payload)
            if payload["decision"] == "no_change":
                return payload
            if payload["decision"] == "full_rebuild":
                return payload
            tree_payload = payload.get("navigation_tree")
            if not isinstance(tree_payload, dict):
                raise ValueError("decision 不是 no_change/full_rebuild 时必须提供 navigation_tree")
            candidate_tree = _coerce_topology_navigation_tree(
                tree_payload,
                facts={"source_digest": "topology-incremental"},
                active_slugs=None,
            )
            candidate_tree = _normalize_navigation_tree_module_slugs(candidate_tree)
            _assert_navigation_tree_not_fallback_only(candidate_tree)
            payload_issues = _collect_navigation_payload_issues(candidate_tree, entries_map)
            if (
                payload_issues["unassigned_paths"]
                or payload_issues["duplicate_paths"]
                or payload_issues["invalid_paths"]
            ):
                raise ValueError(_build_navigation_repair_feedback(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    issues=payload_issues,
                ))
            payload["navigation_tree"] = candidate_tree
            return payload
        except Exception as exc:
            last_exc = exc
            messages.append({
                "role": "user",
                "content": (
                    f"上一次增量导航判定未通过（第 {attempt}/{max_attempts} 次）。"
                    f"错误：{_format_exception_brief(exc)}\n"
                    "请保持当前 tree 稳定，只做更小范围修正；若局部修正不可信，请直接提升为 area_rebuild 或 full_rebuild。"
                ),
            })
    assert last_exc is not None
    raise last_exc


async def _build_sections_from_tree_preserving_existing(
    *,
    root_path: Path,
    tree: NavigationTree,
    entries_map: dict[str, IndexEntry],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, list[IndexEntry]]], set[str]]:
    raw_sections = _build_sections_from_navigation_tree(tree, entries_map)
    recovered_sections = {
        section["slug"]: section
        for section in await _load_sections_from_existing_module_docs(root_path)
    }
    merged_sections: list[dict[str, Any]] = []
    new_slugs: set[str] = set()

    for raw_section in raw_sections:
        slug = raw_section["slug"]
        existing = recovered_sections.get(slug)
        if existing is None:
            section = {
                "name": raw_section["name"],
                "slug": slug,
                "file_paths": list(raw_section.get("file_paths", [])),
                "body_lines": list(raw_section.get("body_lines", [])),
            }
            new_slugs.add(slug)
        else:
            section = {
                "name": existing.get("name", raw_section["name"]),
                "slug": slug,
                "file_paths": list(existing.get("file_paths", [])),
                "body_lines": list(existing.get("body_lines", [])),
            }
            section["name"] = raw_section["name"]
            _update_section_file_list(section, list(raw_section.get("file_paths", [])))
            _update_section_module_link(section, slug=slug)
        merged_sections.append(section)

    valid_sections, module_specs = _prepare_module_specs(merged_sections, entries_map)
    return valid_sections, module_specs, new_slugs


async def _update_annotations_incremental(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    *,
    changed_files: list[str],
    deleted_files: list[str],
    model: str | None = None,
    serena_client: SerenaClient | None = None,
) -> None:
    """增量更新树状导航。

    新链路：
    - 仅内容变化：跳过导航更新
    - 结构变化：交给轻量 navigation agent 判断
    - `no_change`：不动导航层
    - `module_update` / `area_rebuild`：提交完整 tree 快照，仅重渲染导航层；modules 正文尽量保留
    - `full_rebuild`：回落到全量 topology 链路
    """
    modules_dir = _annotation_modules_dir(root_path)
    entries_map = {str(e.file_meta.path): e for e in entries}

    tree_path = _navigation_tree_path(root_path)
    try:
        raw_tree = await asyncio.to_thread(tree_path.read_text, "utf-8")
        tree_payload = json.loads(raw_tree)
        expected_version = str(NavigationTree.model_fields["version"].default)
        actual_version = str(tree_payload.get("version") or "").strip()
        if actual_version != expected_version:
            logger.info(
                "现有 navigation_tree 版本过期(current=%s, expected=%s)，降级为全量认知文档重建",
                actual_version or "(missing)",
                expected_version,
            )
            await _write_annotations(
                entries,
                project_meta,
                root_path,
                model=model,
                serena_client=serena_client,
            )
            return
        tree = NavigationTree.model_validate(tree_payload)
    except Exception:
        logger.info(
            "未找到可用 navigation_tree.json，降级为全量认知文档重建"
        )
        await _write_annotations(
            entries,
            project_meta,
            root_path,
            model=model,
            serena_client=serena_client,
        )
        return

    if serena_client is None:
        logger.info("增量导航缺少 serena_client，降级为全量认知文档重建")
        await _write_annotations(
            entries,
            project_meta,
            root_path,
            model=model,
            serena_client=serena_client,
        )
        return

    decision_payload = await _run_incremental_navigation_update(
        entries=entries,
        project_meta=project_meta,
        root_path=root_path,
        tree=tree,
        changed_files=changed_files,
        deleted_files=deleted_files,
        model=model,
        serena_client=serena_client,
    )

    decision = str(decision_payload.get("decision") or "").strip()
    if decision == "no_change":
        logger.info("增量导航判定: no_change，跳过导航层更新")
        return

    if decision == "full_rebuild":
        logger.info("增量导航判定: full_rebuild，降级为全量认知文档重建")
        await _write_annotations(
            entries,
            project_meta,
            root_path,
            model=model,
            serena_client=serena_client,
        )
        return

    candidate_tree = decision_payload.get("navigation_tree")
    if not isinstance(candidate_tree, NavigationTree):
        raise RuntimeError("增量导航更新缺少有效 navigation_tree")

    valid_sections, module_specs, new_slugs = await _build_sections_from_tree_preserving_existing(
        root_path=root_path,
        tree=candidate_tree,
        entries_map=entries_map,
    )
    sections_by_slug = {section["slug"]: section for section in valid_sections}
    active_slugs = {slug for _, slug, _ in module_specs}
    stable_tree = _repair_navigation_tree(
        candidate_tree.model_copy(update={
            "generated_at": datetime.now(UTC),
            "source_digest": _compute_source_digest(valid_sections),
        }),
        active_slugs,
    )
    stable_tree = _sync_tree_modules_from_sections(stable_tree, sections_by_slug)
    _assert_navigation_tree_not_fallback_only(stable_tree)
    tree_errors = _validate_navigation_tree(stable_tree, active_slugs)
    if tree_errors:
        raise RuntimeError("增量导航产物校验失败: " + " | ".join(tree_errors))
    await _write_empty_module_skeletons(
        [spec for spec in module_specs if spec[1] in new_slugs],
        modules_dir,
    )
    await _persist_annotation_outputs(
        root_path=root_path,
        tree=stable_tree,
        sections_by_slug=sections_by_slug,
        module_specs=module_specs,
    )

    logger.info(
        "认知导航增量更新完成: decision=%s areas=%d modules=%d new_modules=%d",
        decision,
        len(stable_tree.areas),
        len(module_specs),
        len(new_slugs),
    )


# ============================================================================
# 核心索引逻辑
# ============================================================================
