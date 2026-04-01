"""Digest 两阶段轻量特化 Agent。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base_agent import (
    BaseReActAgent,
    DELIVER_TOOL,
    DeliverDecision,
    LoopState,
    REACT_LENGTH_EXHAUSTED,
    REACT_LLM_EXHAUSTED,
    REACT_NO_TOOL_EXHAUSTED,
    REACT_TIMEOUT,
    REACT_TIMEOUT_BUDGET,
    _extract_tool_call_args,
    _get_tool_call_id,
    _get_tool_name,
    _safe_json_dumps,
)
from .file_discovery import first_blocked_tool_path
from .memory import load_file_baseline
from .models import ChangedFileFact, InsightFact, NavigationTree
from .prompt_guard import fit_text_to_budget
from .serena_client import SerenaClient

logger = logging.getLogger(__name__)
_DEFAULT_MAX_SECONDS = 180.0
_FILTER_MIN_BUDGET = 3
_FILTER_MAX_BUDGET = 50
_FILTER_FACTOR = 3
_ASSIMILATION_MIN_BUDGET = 10
_ASSIMILATION_MAX_BUDGET = 75
_ASSIMILATION_FACTOR = 5
_FACTS_NOTICE = "\n... [facts 已截断，可调用 read_facts 读取完整版本] ...\n"
_FAILURE_MESSAGES: dict[str, str] = {
    REACT_TIMEOUT_BUDGET: "Digest 特化 Agent ReAct 循环超时，已达最大时间预算",
    REACT_NO_TOOL_EXHAUSTED: "Digest 特化 Agent 连续未产生 tool_calls，已终止",
    REACT_LENGTH_EXHAUSTED: "Digest 特化 Agent 输出连续被截断，已放弃本轮",
    REACT_TIMEOUT: "Digest 特化 Agent 模型调用超时，重试次数耗尽",
    REACT_LLM_EXHAUSTED: "Digest 特化 Agent 模型降级链耗尽，已终止",
}


def _annotation_root_dir(root: Path) -> Path:
    return root / ".pce" / "annotations"


def _annotation_index_path(root: Path) -> Path:
    return _annotation_root_dir(root) / "index.md"


def _annotation_areas_dir(root: Path) -> Path:
    return _annotation_root_dir(root) / "areas"


def _annotation_modules_dir(root: Path) -> Path:
    return _annotation_root_dir(root) / "modules"


def _normalize_rel_path(path: str) -> str:
    text = Path(path).as_posix().replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text


def _annotation_path_allowed(path: str) -> bool:
    normalized = _normalize_rel_path(path)
    if normalized == ".pce/annotations/index.md":
        return True
    return (
        normalized.startswith(".pce/annotations/areas/")
        or normalized.startswith(".pce/annotations/modules/")
    ) and normalized.endswith(".md")


def _iter_available_annotation_paths(project_root: Path) -> list[str]:
    paths: list[str] = []
    index_path = _annotation_index_path(project_root)
    if index_path.exists():
        paths.append(".pce/annotations/index.md")
    for base in (_annotation_areas_dir(project_root), _annotation_modules_dir(project_root)):
        if not base.exists():
            continue
        for path in sorted(base.glob("*.md")):
            paths.append(path.relative_to(project_root).as_posix())
    return paths


async def _read_text_if_exists(path: Path) -> str:
    try:
        return await asyncio.to_thread(path.read_text, "utf-8")
    except FileNotFoundError:
        return ""


async def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    await asyncio.to_thread(tmp_path.write_text, content, "utf-8")
    try:
        await asyncio.to_thread(os.replace, tmp_path, path)
    finally:
        if tmp_path.exists():
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)


def _truncate_block(text: str, *, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 20
    return f"{text[:head].rstrip()}\n... [truncated] ...\n{text[-tail:].lstrip()}".strip()


def _render_insight(insight: InsightFact) -> str:
    return "\n".join(
        [
            f"- id: `{insight.id}`",
            f"  - created_at: `{insight.created_at.isoformat()}`",
            f"  - confidence: `{insight.confidence}`",
            f"  - question: {_truncate_block(insight.question, max_chars=400)}",
            f"  - answer: {_truncate_block(insight.answer, max_chars=800)}",
        ]
    )


def _format_patch_blocks(file_fact: ChangedFileFact, *, limit: int = 3) -> str:
    if not file_fact.patch_blocks:
        if file_fact.status == "created" and file_fact.new_content:
            return f"```diff\n+ file created\n{_truncate_block(file_fact.new_content, max_chars=900)}\n```"
        if file_fact.status == "deleted" and file_fact.old_content:
            return f"```diff\n- file deleted\n{_truncate_block(file_fact.old_content, max_chars=900)}\n```"
        return ""
    chunks: list[str] = []
    for block in file_fact.patch_blocks[:limit]:
        old = _truncate_block(block.old_snippet or "", max_chars=500)
        new = _truncate_block(block.new_snippet or "", max_chars=500)
        lines = ["```diff"]
        if block.old_start and block.new_start:
            lines.append(
                f"@@ -{block.old_start},{(block.old_end or block.old_start) - block.old_start + 1} "
                f"+{block.new_start},{(block.new_end or block.new_start) - block.new_start + 1} @@"
            )
        if old:
            lines.extend(f"- {line}" for line in old.splitlines())
        if new:
            lines.extend(f"+ {line}" for line in new.splitlines())
        lines.append("```")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _render_patch_fact(file_fact: ChangedFileFact) -> str:
    patch = _format_patch_blocks(file_fact)
    if not patch:
        return f"### `{file_fact.path}` [{file_fact.status}]"
    return f"### `{file_fact.path}` [{file_fact.status}]\n{patch}"


def _summarize_tool_args(args: dict[str, Any], *, limit: int = 160) -> str:
    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            preview = value
        elif isinstance(value, list):
            preview = f"list[{len(value)}]"
        elif isinstance(value, dict):
            preview = f"dict[{len(value)}]"
        else:
            preview = str(value)
        if len(preview) > 60:
            preview = preview[:57].rstrip() + "..."
        parts.append(f"{key}={preview}")
    text = ", ".join(parts)
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def build_filter_facts_text(
    *,
    insights: list[InsightFact],
    annotation_paths: list[str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是 digest stageA 直接注入的 insights。你的任务只是筛选哪些值得进入下一阶段。",
        "",
        "## Insights",
    ]
    lines.extend(_render_insight(item) for item in insights)
    lines.extend([
        "",
        "## Readable Annotations",
    ])
    lines.extend([f"- `{item}`" for item in annotation_paths] or ["- (none)"])
    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=32000,
        notice=_FACTS_NOTICE,
        min_chars=4000,
    )
    return fitted, fitted != full


def build_assimilation_facts_text(
    *,
    insights: list[InsightFact],
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    annotation_paths: list[str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是 digest stageB 直接注入的 facts。目标是在有限预算下尽量完成认知内化。",
        "",
        "## Insights",
    ]
    lines.extend(_render_insight(item) for item in insights)
    lines.extend(["", "## Dirty Files"])
    lines.extend([f"- `{item}`" for item in dirty_files] or ["- (none)"])
    lines.extend(["", "## Patch Evidence"])
    if patch_facts:
        lines.extend(_render_patch_fact(item) for item in patch_facts)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Readable Annotations"])
    lines.extend([f"- `{item}`" for item in annotation_paths] or ["- (none)"])
    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=32000,
        notice=_FACTS_NOTICE,
        min_chars=5000,
    )
    return fitted, fitted != full


def build_audit_facts_text(
    *,
    insights: list[InsightFact],
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    annotation_paths: list[str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是 digest stageAC 直接注入的 facts。目标是先保证现有认知不过期，再保守筛出值得进入内化阶段的 insight。",
        "",
        "## Insights",
    ]
    lines.extend(_render_insight(item) for item in insights or [])
    if not insights:
        lines.append("- (none)")
    lines.extend(["", "## Dirty Files"])
    lines.extend([f"- `{item}`" for item in dirty_files] or ["- (none)"])
    lines.extend(["", "## Diff Evidence (excerpt)"])
    if patch_facts:
        lines.extend(_render_patch_fact(item) for item in patch_facts)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Readable Annotations"])
    lines.extend([f"- `{item}`" for item in annotation_paths] or ["- (none)"])
    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=32000,
        notice=_FACTS_NOTICE,
        min_chars=5000,
    )
    return fitted, fitted != full


def _compute_filter_budget(insight_count: int) -> int:
    raw = insight_count * _FILTER_FACTOR
    return max(_FILTER_MIN_BUDGET, min(_FILTER_MAX_BUDGET, raw))


def _compute_assimilation_budget(insight_count: int) -> int:
    raw = insight_count * _ASSIMILATION_FACTOR
    return max(_ASSIMILATION_MIN_BUDGET, min(_ASSIMILATION_MAX_BUDGET, raw))


@dataclass
class FilterDecision:
    keep_insight_ids: list[str]
    drop_insight_ids: list[str]
    notes: list[str]


@dataclass
class SharedToolBudget:
    total: int
    used: int = 0


@dataclass
class AssimilationResult:
    summary: str


@dataclass
class CleanupResult:
    summary: str


@dataclass
class AuditResult:
    keep_insight_ids: list[str]
    drop_insight_ids: list[str]
    notes: list[str]
    summary: str


class _DigestStageAgent(BaseReActAgent):
    _temperature_env_key = "PCE_DIGEST_TEMPERATURE"
    _completion_temperature = 0.1
    _enable_budget_warning = False

    def __init__(
        self,
        *,
        project_root: Path,
        facts_text: str,
        facts_truncated: bool,
        tool_budget: int,
        shared_budget: SharedToolBudget | None,
        allow_serena: bool,
        allow_read_annotation: bool = True,
        model: str | None = None,
        provider: str | None = None,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
    ) -> None:
        super().__init__(model=model, provider=provider, max_seconds=max_seconds)
        self._project_root = project_root.resolve()
        self._facts_text = facts_text.strip()
        self._facts_truncated = facts_truncated
        self._shared_budget = shared_budget
        self._tool_budget_total = shared_budget.total if shared_budget is not None else tool_budget
        self._tool_budget_used = shared_budget.used if shared_budget is not None else 0
        self._budget_exhausted_notified = False
        self._deliver_tool = DELIVER_TOOL
        self._virtual_tool_names_set: set[str] = set()
        self._allow_serena = allow_serena
        self._allow_read_annotation = allow_read_annotation
        self._annotation_paths = _iter_available_annotation_paths(self._project_root)
        self._stage_started_at = 0.0
        self._stage_name = self.__class__.__name__

    def _budget_note(self) -> str:
        remaining = max(0, self._tool_budget_total - self._tool_budget_used)
        return (
            f"\n\n[预算提示] 已使用 {self._tool_budget_used}/{self._tool_budget_total} 次工具预算，"
            f"剩余 {remaining} 次。请优先高价值、窄范围探索。"
        )

    async def on_before_round(self, state: LoopState) -> None:
        if self._stage_started_at <= 0:
            self._stage_started_at = time.monotonic()
        self._sync_budget_from_shared()
        logger.info(
            "%s round=%d budget=%d/%d remaining=%.1fs",
            self._stage_name,
            state.round_num,
            self._tool_budget_used,
            self._tool_budget_total,
            state.remaining,
        )
        if (
            self._tool_budget_total > 0
            and self._tool_budget_used >= self._tool_budget_total
            and not self._budget_exhausted_notified
        ):
            self._append(state, {"role": "user", "content": self._build_budget_exhausted_prompt()})
            self._budget_exhausted_notified = True

    def build_tools_schema(self, serena_client: SerenaClient, state: LoopState) -> list[dict[str, Any]]:
        self._sync_budget_from_shared()
        tools: list[dict[str, Any]] = []
        if self._allow_serena and self._tool_budget_used < self._tool_budget_total:
            for schema in serena_client.tools_schema:
                try:
                    name = schema["function"]["name"]
                except Exception:
                    continue
                tools.append(schema)
        virtual = self._build_virtual_tools()
        self._virtual_tool_names_set = {schema["function"]["name"] for schema in virtual}
        return tools + virtual + [self._deliver_tool]

    @property
    def virtual_tool_names(self) -> set[str]:
        return self._virtual_tool_names_set

    async def _invoke_serena(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
        state: LoopState | None = None,
    ) -> dict[str, Any]:
        tool_name = _get_tool_name(tool_call) or ""
        tool_args = _extract_tool_call_args(tool_call) or {}
        blocked_reason = self._blocked_serena_access_reason(tool_name, tool_args)
        if blocked_reason is not None:
            logger.info(
                "%s tool=serena/%s blocked args=[%s]",
                self._stage_name,
                tool_name or "unknown",
                _summarize_tool_args(tool_args),
            )
            return {
                "tool_call_id": _get_tool_call_id(tool_call),
                "name": tool_name or "unknown",
                "content": blocked_reason,
            }
        logger.info(
            "%s tool=serena/%s start args=[%s]",
            self._stage_name,
            tool_name or "unknown",
            _summarize_tool_args(tool_args),
        )
        result = await super()._invoke_serena(tool_call, serena_client, state=state)
        if self._tool_budget_used < self._tool_budget_total:
            self._consume_budget()
            result["content"] = str(result.get("content") or "") + self._budget_note()
        logger.info(
            "%s tool=serena/%s done budget=%d/%d result_chars=%d",
            self._stage_name,
            tool_name or "unknown",
            self._tool_budget_used,
            self._tool_budget_total,
            len(str(result.get("content") or "")),
        )
        return result

    def _blocked_serena_access_reason(self, tool_name: str, tool_args: dict[str, Any]) -> str | None:
        del tool_name
        blocked = first_blocked_tool_path(self._project_root, tool_args)
        if blocked is None:
            return None
        normalized, access = blocked
        if access == "virtual_only":
            return (
                f"路径 `{normalized}` 属于受控内部认知路径。"
                "这类路径不能通过 Serena 访问；如需读取 annotation，请使用 `read_annotation`。"
            )
        return (
            f"路径 `{normalized}` 不在当前 digest agent 的可读边界内。"
            "该路径命中了项目 ignore / PCE ignore / 安全围栏，因此不可通过 Serena 读取。"
        )

    def _build_base_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "system", "content": self._build_context_boundary_prompt()},
            {"role": "system", "content": self._facts_text},
        ]

    def _build_context_boundary_prompt(self) -> str:
        lines = [
            "## 认知读取边界",
            "- `Readable Annotations` 中列出的 annotation 只能通过 `read_annotation` 访问。",
            "- stageAC / stageB 默认不读代码；若允许读取 diff，只能通过受控 diff 工具访问。",
            "- digest 当前不应依赖 Serena 做代码探索；若已注入 facts 足够，请直接基于 annotation 与 diff 收口。",
            "- 若已注入 facts 足够，请不要为了完整性继续探索。",
        ]
        return "\n".join(lines)

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if self._facts_truncated:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_facts",
                    "description": "读取未截断的完整 facts。仅在当前注入 facts 不足时使用。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
        if self._allow_read_annotation:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_annotation",
                    "description": "读取指定 annotation 文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            })
        tools.extend(self._extra_virtual_tools())
        return tools

    async def handle_virtual_tool(self, tool_call: Any, state: LoopState) -> dict[str, Any] | None:
        tc_id = _get_tool_call_id(tool_call)
        name = _get_tool_name(tool_call) or "unknown"
        args = _extract_tool_call_args(tool_call) or {}

        def _ok(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        def _err(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        if name == "read_facts":
            logger.info("%s tool=virtual/read_facts", self._stage_name)
            return _ok(self._facts_text.replace(_FACTS_NOTICE, ""))
        if name == "read_annotation":
            if not self._allow_read_annotation:
                return _err("当前阶段不允许读取 annotation")
            path = str(args.get("path") or "").strip()
            if not _annotation_path_allowed(path):
                return _err("annotation path 不合法")
            self._consume_budget()
            content = await _read_text_if_exists(self._project_root / _normalize_rel_path(path))
            logger.info(
                "%s tool=virtual/read_annotation path=%s budget=%d/%d chars=%d",
                self._stage_name,
                path,
                self._tool_budget_used,
                self._tool_budget_total,
                len(content),
            )
            return _ok(content)
        return await self._handle_stage_virtual_tool(name, args, _ok, _err)

    def _sync_budget_from_shared(self) -> None:
        if self._shared_budget is not None:
            self._tool_budget_total = self._shared_budget.total
            self._tool_budget_used = self._shared_budget.used

    def _consume_budget(self, amount: int = 1) -> None:
        self._tool_budget_used += amount
        if self._shared_budget is not None:
            self._shared_budget.used = self._tool_budget_used

    def build_no_tool_correction(self, state: LoopState, finish_reason: str) -> dict[str, Any]:
        return {"role": "user", "content": self._build_no_tool_prompt()}

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return []

    async def _handle_stage_virtual_tool(
        self,
        name: str,
        args: dict[str, Any],
        ok_builder: Any,
        err_builder: Any,
    ) -> dict[str, Any]:
        return err_builder(f"未知虚拟工具: {name}")

    def _build_budget_exhausted_prompt(self) -> str:
        raise NotImplementedError

    def _build_no_tool_prompt(self) -> str:
        raise NotImplementedError

    def _build_system_prompt(self) -> str:
        raise NotImplementedError


def _build_file_diff_excerpt(
    *,
    path: str,
    old_content: str | None,
    new_content: str | None,
    max_lines: int = 50,
) -> str:
    import difflib

    old_lines = (old_content or "").splitlines()
    new_lines = (new_content or "").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return f"### `{path}`\n```diff\n(no diff)\n```"
    excerpt = diff_lines[:max_lines]
    return f"### `{path}`\n```diff\n" + "\n".join(excerpt) + "\n```"


def _split_markdown_sections(raw: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    header_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current_heading: str | None = None
    current_body: list[str] = []
    seen_section = False

    def _flush() -> None:
        nonlocal current_heading, current_body
        if current_heading is None:
            return
        sections[current_heading] = current_body[:]
        if current_heading not in order:
            order.append(current_heading)

    for raw_line in raw.splitlines():
        if raw_line.startswith("## "):
            seen_section = True
            _flush()
            current_heading = raw_line[3:].strip()
            current_body = []
            continue
        if not seen_section:
            header_lines.append(raw_line)
        elif current_heading is not None:
            current_body.append(raw_line.rstrip())
    _flush()
    return header_lines, sections, order


def _render_markdown_sections(header_lines: list[str], sections: dict[str, list[str]], order: list[str]) -> str:
    lines = [line.rstrip() for line in header_lines]
    while lines and not lines[-1].strip():
        lines.pop()
    for heading in order:
        body = sections.get(heading)
        if body is None:
            continue
        lines.extend(["", f"## {heading}"])
        lines.extend(body)
    return "\n".join(lines).rstrip() + "\n"


def _split_markdown_title_block(raw: str) -> tuple[str, list[str], dict[str, list[str]], list[str]]:
    lines = raw.splitlines()
    title = ""
    body_before_sections: list[str] = []
    section_lines: list[str] = []
    seen_sections = False

    for line in lines:
        if not title and line.startswith("# "):
            title = line.rstrip()
            continue
        if line.startswith("## "):
            seen_sections = True
        if seen_sections:
            section_lines.append(line)
        else:
            body_before_sections.append(line.rstrip())

    _, sections, order = _split_markdown_sections("\n".join(section_lines))
    return title or "#", body_before_sections, sections, order


def _render_markdown_title_block(
    title: str,
    title_body: list[str],
    sections: dict[str, list[str]],
    order: list[str],
) -> str:
    lines = [title.rstrip()]
    cleaned_title_body = [line.rstrip() for line in title_body]
    while cleaned_title_body and not cleaned_title_body[0].strip():
        cleaned_title_body.pop(0)
    while cleaned_title_body and not cleaned_title_body[-1].strip():
        cleaned_title_body.pop()
    if cleaned_title_body:
        lines.extend(["", *cleaned_title_body])
    seen_headings: set[str] = set()
    for heading in order:
        if heading in seen_headings:
            continue
        seen_headings.add(heading)
        body = sections.get(heading)
        if body is None:
            continue
        lines.extend(["", f"## {heading}"])
        lines.extend(body)
    return "\n".join(lines).rstrip() + "\n"


async def _build_annotation_skeleton(project_root: Path, path: str) -> str:
    from .annotation_writer import (
        _annotation_index_path,
        _build_empty_module_skeleton_md,
        _build_sections_from_navigation_tree,
        _navigation_tree_path,
        _render_area_md,
        _render_hierarchical_index_md,
    )
    from .memory import load_index

    normalized = _normalize_rel_path(path)
    tree_path = _navigation_tree_path(project_root)
    raw_tree = await _read_text_if_exists(tree_path)
    if not raw_tree:
        raise ValueError("navigation_tree 不存在，无法重置骨架")
    tree = NavigationTree.model_validate(json.loads(raw_tree))
    snapshot = await load_index(root_path=project_root)
    if snapshot is None:
        raise ValueError("index snapshot 不存在，无法重置骨架")
    entries_map = {str(entry.file_meta.path): entry for entry in snapshot.entries}
    sections = _build_sections_from_navigation_tree(tree, entries_map)
    sections_by_slug = {section["slug"]: section for section in sections}

    if normalized == _annotation_index_path(project_root).relative_to(project_root).as_posix():
        return _render_hierarchical_index_md(tree, sections_by_slug)

    if normalized.startswith(".pce/annotations/areas/") and normalized.endswith(".md"):
        slug = Path(normalized).stem
        area = next((item for item in tree.areas if item.slug == slug), None)
        if area is None:
            raise ValueError("area 不存在，无法重置骨架")
        return _render_area_md(area, sections_by_slug)

    if normalized.startswith(".pce/annotations/modules/") and normalized.endswith(".md"):
        slug = Path(normalized).stem
        section = sections_by_slug.get(slug)
        if section is None:
            raise ValueError("module 不存在，无法重置骨架")
        module_name = str(section.get("name") or slug)
        file_paths = list(section.get("file_paths", []))
        entries = [entries_map[item] for item in file_paths if item in entries_map]
        return _build_empty_module_skeleton_md(module_name, entries)

    raise ValueError("annotation path 不支持骨架重置")


class DigestAssimilationStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        insight_count: int,
        facts_text: str,
        facts_truncated: bool,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=_compute_assimilation_budget(insight_count),
            shared_budget=None,
            allow_serena=False,
            model=model,
            provider=provider,
        )
        self._result: AssimilationResult | None = None

    async def run(self, *, serena_client: SerenaClient) -> AssimilationResult:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stageB。请基于保留下来的 insights、dirty files 和 patch evidence，"
                "在有限预算下做去重后的有限增量认知内化。只允许读取 annotation，并只改写已有 `#` / `##` 级内容。"
            ),
        })
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._result is None:
            raise RuntimeError("stageB 未提交结果")
        return self._result

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestAssimilationAgent，负责将已筛选保留的 insight 做尽量内化。",
                "",
                "## 目标",
                "- 在有限预算下尽量把 insight 沉淀进合适的 annotation 文档。",
                "- 仅做有限增量内化；若模棱两可，宁可漏掉也不要多写。",
                "- 若核验后认为与现有认知重复，或无需改动，也可以直接结束本轮。",
                "",
                "## 约束",
                "- 优先依赖直接注入的 insights、dirty files、patch evidence。",
                "- 只允许读取 `Readable Annotations` 中的 annotation，不做 Serena 代码探索。",
                "- 只允许改写已存在的 `#` 顶层正文或已存在的 `##` section 正文；不得新增标题，不得整篇覆写。",
                "- 由你自行判断应修改哪些 annotation 文件；Python 不会预先决定层级。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若当前 facts 已足够，请直接按已有 `#` / `##` section 做有限改写（如需要）并调用 `deliver_stage_result` 收口；"
            "若模棱两可或疑似重复，宁可跳过。不要自然语言总结。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return "stageB 工具预算已耗尽。请停止探索，基于当前证据直接收口。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._result is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交极简结果，再 deliver。",
            })
        logger.info(
            "%s deliver elapsed=%.2fs rounds=%d summary_chars=%d",
            self._stage_name,
            state.elapsed,
            state.round_num,
            len(self._result.summary),
        )
        return DeliverDecision.finish(_safe_json_dumps({"summary": self._result.summary}))

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "rewrite_sections",
                    "description": "按已存在的 `#` / `##` 标题重写 annotation 中对应正文。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["heading", "content"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["path", "sections"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交 stageB 的极简完成说明。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        ]

    async def _handle_stage_virtual_tool(
        self,
        name: str,
        args: dict[str, Any],
        ok_builder: Any,
        err_builder: Any,
    ) -> dict[str, Any]:
        if name == "rewrite_sections":
            return await _handle_rewrite_sections_tool(
                project_root=self._project_root,
                args=args,
                ok_builder=ok_builder,
                err_builder=err_builder,
                budget_callback=self._consume_budget,
                budget_note_builder=self._budget_note,
            )
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        self._result = AssimilationResult(summary=str(args.get("summary") or "").strip())
        logger.info(
            "%s deliver_stage_result summary_chars=%d",
            self._stage_name,
            len(self._result.summary),
        )
        return ok_builder("stageB 结果已接收")


async def _handle_rewrite_sections_tool(
    *,
    project_root: Path,
    args: dict[str, Any],
    ok_builder: Any,
    err_builder: Any,
    budget_callback: Any,
    budget_note_builder: Any,
) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    raw_sections = args.get("sections")
    if not _annotation_path_allowed(path):
        return err_builder("annotation path 不合法")
    if not isinstance(raw_sections, list):
        return err_builder("sections 必须是对象数组")
    section_map: dict[str, str] = {}
    for item in raw_sections:
        if not isinstance(item, dict):
            return err_builder("sections 项必须为对象")
        heading = str(item.get("heading") or "").strip()
        content = str(item.get("content") or "")
        if not heading:
            return err_builder("heading 不能为空")
        section_map[heading] = content
    logger.info(
        "rewrite_sections start path=%s headings=%s",
        path,
        ",".join(section_map.keys()),
    )
    existing = await _read_text_if_exists(project_root / _normalize_rel_path(path))
    if not existing:
        return err_builder("annotation 不存在")
    title, title_body, sections, order = _split_markdown_title_block(existing)
    for heading, content in section_map.items():
        body = [line.rstrip() for line in content.splitlines()]
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if heading == "#":
            title_body = body
            continue
        if heading not in sections:
            return err_builder(f"section 不存在: {heading}")
        sections[heading] = body
    await _write_text_atomic(
        project_root / _normalize_rel_path(path),
        _render_markdown_title_block(title, title_body, sections, order),
    )
    budget_callback()
    logger.info("rewrite_sections done path=%s headings=%s", path, ",".join(section_map.keys()))
    return ok_builder(f"sections 已重写: {path}" + budget_note_builder())


async def _handle_reset_annotation_to_skeleton_tool(
    *,
    project_root: Path,
    args: dict[str, Any],
    ok_builder: Any,
    err_builder: Any,
    budget_callback: Any,
    budget_note_builder: Any,
) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    if not _annotation_path_allowed(path):
        return err_builder("annotation path 不合法")
    logger.info("reset_annotation_to_skeleton start path=%s", path)
    skeleton = await _build_annotation_skeleton(project_root, path)
    await _write_text_atomic(project_root / _normalize_rel_path(path), skeleton)
    budget_callback()
    logger.info("reset_annotation_to_skeleton done path=%s", path)
    return ok_builder(f"annotation 已重置为骨架: {path}" + budget_note_builder())


async def _handle_read_diff_tool(
    *,
    project_root: Path,
    dirty_files: set[str],
    args: dict[str, Any],
    ok_builder: Any,
    err_builder: Any,
    budget_callback: Any,
    budget_note_builder: Any,
) -> dict[str, Any]:
    raw_paths = args.get("paths")
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        return err_builder("paths 必须是字符串数组")
    normalized = []
    for item in raw_paths[:10]:
        path = _normalize_rel_path(item)
        if path and path in dirty_files and path not in normalized:
            normalized.append(path)
    if not normalized:
        return err_builder("没有可读取的 dirty file diff")
    logger.info("read_diff start paths=%s", ",".join(normalized))
    budget_callback()
    chunks: list[str] = []
    for path in normalized:
        baseline = await load_file_baseline(path, root_path=project_root)
        old_content = baseline.content if baseline is not None else None
        abs_path = project_root / path
        new_content = await _read_text_if_exists(abs_path) if abs_path.exists() else None
        chunks.append(_build_file_diff_excerpt(path=path, old_content=old_content, new_content=new_content, max_lines=5000))
    logger.info("read_diff done paths=%s", ",".join(normalized))
    return ok_builder("\n\n".join(chunks) + budget_note_builder())


class DigestAuditStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        insights: list[InsightFact],
        dirty_files: list[str],
        facts_text: str,
        facts_truncated: bool,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=max(_compute_filter_budget(len(insights)), len(dirty_files) * 3),
            shared_budget=None,
            allow_serena=False,
            allow_read_annotation=True,
            model=model,
            provider=provider,
        )
        self._result: AuditResult | None = None
        self._dirty_files = {str(item).strip() for item in dirty_files if str(item).strip()}

    async def run(self, *, serena_client: SerenaClient) -> AuditResult:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stageAC。请先基于 annotation 与 diff 判断哪些 insight 低价值、重复或存在明显陈旧风险，"
                "并清理已有 annotation 中可能过期的内容。模棱两可时，宁可漏掉、不多写；宁可不清，也不要误清。"
            ),
        })
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._result is None:
            raise RuntimeError("stageAC 未提交结果")
        return self._result

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestAuditAgent，负责先保证现有认知不过期，再保守筛选值得进入内化阶段的 insight。",
                "",
                "## 目标",
                "- 识别并清理因 dirty files 可能已陈旧的 annotation。",
                "- 过滤低价值、明显重复、或已明显陈旧的 insight。",
                "- 若判断模棱两可，宁可漏掉，不要多写、不要误清。",
                "",
                "## 约束",
                "- 只允许读取 annotation 和 dirty file diff；不要做 Serena 代码探索。",
                "- 清理动作仅限：局部重置为骨架，或按已有 `#` / `##` 标题重写正文。",
                "- 可同时完成 insight 筛选与 annotation 清理；即使某 insight 仍存在，也不妨碍你先清理可能误导的旧认知。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "若当前 facts 已足够，请直接提交 keep/drop 结果；若确需判断陈旧或重复，只读取 annotation 或 diff。"
            "模棱两可时，宁可保守跳过。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return "stageAC 工具预算已耗尽。请停止探索，基于当前证据保守收口。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._result is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交 stageAC 结果，再 deliver。",
            })
        logger.info(
            "%s deliver elapsed=%.2fs rounds=%d keep=%d drop=%d",
            self._stage_name,
            state.elapsed,
            state.round_num,
            len(self._result.keep_insight_ids),
            len(self._result.drop_insight_ids),
        )
        return DeliverDecision.finish(
            _safe_json_dumps(
                {
                    "keep_insight_ids": self._result.keep_insight_ids,
                    "drop_insight_ids": self._result.drop_insight_ids,
                    "notes": self._result.notes,
                    "summary": self._result.summary,
                }
            )
        )

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_diff",
                    "description": "读取 dirty file 的完整 baseline diff。",
                    "parameters": {
                        "type": "object",
                        "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                        "required": ["paths"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rewrite_sections",
                    "description": "按已存在的 `#` / `##` 标题重写 annotation 中的正文。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["heading", "content"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["path", "sections"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reset_annotation_to_skeleton",
                    "description": "将指定 annotation 局部重置为来自当前 navigation_tree 的骨架。",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交 stageAC 的筛选与清理结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keep_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "drop_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "notes": {"type": "array", "items": {"type": "string"}},
                            "summary": {"type": "string"},
                        },
                        "required": ["keep_insight_ids", "drop_insight_ids"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def _handle_stage_virtual_tool(
        self,
        name: str,
        args: dict[str, Any],
        ok_builder: Any,
        err_builder: Any,
    ) -> dict[str, Any]:
        if name == "read_diff":
            return await _handle_read_diff_tool(
                project_root=self._project_root,
                dirty_files=self._dirty_files,
                args=args,
                ok_builder=ok_builder,
                err_builder=err_builder,
                budget_callback=self._consume_budget,
                budget_note_builder=self._budget_note,
            )
        if name == "rewrite_sections":
            return await _handle_rewrite_sections_tool(
                project_root=self._project_root,
                args=args,
                ok_builder=ok_builder,
                err_builder=err_builder,
                budget_callback=self._consume_budget,
                budget_note_builder=self._budget_note,
            )
        if name == "reset_annotation_to_skeleton":
            return await _handle_reset_annotation_to_skeleton_tool(
                project_root=self._project_root,
                args=args,
                ok_builder=ok_builder,
                err_builder=err_builder,
                budget_callback=self._consume_budget,
                budget_note_builder=self._budget_note,
            )
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        keep_ids = args.get("keep_insight_ids")
        drop_ids = args.get("drop_insight_ids")
        notes = args.get("notes") or []
        summary = str(args.get("summary") or "").strip()
        if not isinstance(keep_ids, list) or not all(isinstance(item, str) for item in keep_ids):
            return err_builder("keep_insight_ids 必须是字符串数组")
        if not isinstance(drop_ids, list) or not all(isinstance(item, str) for item in drop_ids):
            return err_builder("drop_insight_ids 必须是字符串数组")
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            return err_builder("notes 必须是字符串数组")
        self._result = AuditResult(
            keep_insight_ids=list(dict.fromkeys(item.strip() for item in keep_ids if item.strip())),
            drop_insight_ids=list(dict.fromkeys(item.strip() for item in drop_ids if item.strip())),
            notes=[item.strip() for item in notes if item.strip()],
            summary=summary,
        )
        logger.info(
            "%s deliver_stage_result keep=%d drop=%d summary_chars=%d",
            self._stage_name,
            len(self._result.keep_insight_ids),
            len(self._result.drop_insight_ids),
            len(self._result.summary),
        )
        return ok_builder("stageAC 结果已接收")


async def run_digest_audit(
    *,
    project_root: Path,
    insights: list[InsightFact],
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
) -> AuditResult:
    annotation_paths = _iter_available_annotation_paths(project_root)
    facts_text, truncated = build_audit_facts_text(
        insights=insights,
        dirty_files=dirty_files,
        patch_facts=patch_facts,
        annotation_paths=annotation_paths,
        model=model or "gpt-4o-mini",
    )
    agent = DigestAuditStageAgent(
        project_root=project_root,
        insights=insights,
        dirty_files=dirty_files,
        facts_text=facts_text,
        facts_truncated=truncated,
        model=model,
        provider=provider,
    )
    return await agent.run(serena_client=serena_client)


async def run_digest_assimilation(
    *,
    project_root: Path,
    insights: list[InsightFact],
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
) -> AssimilationResult:
    annotation_paths = _iter_available_annotation_paths(project_root)
    facts_text, truncated = build_assimilation_facts_text(
        insights=insights,
        dirty_files=dirty_files,
        patch_facts=patch_facts,
        annotation_paths=annotation_paths,
        model=model or "gpt-4o-mini",
    )
    agent = DigestAssimilationStageAgent(
        project_root=project_root,
        insight_count=len(insights),
        facts_text=facts_text,
        facts_truncated=truncated,
        model=model,
        provider=provider,
    )
    return await agent.run(serena_client=serena_client)
