"""Digest 两阶段轻量特化 Agent。"""

from __future__ import annotations

import asyncio
import json
import os
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

_DEFAULT_MAX_SECONDS = 180.0
_FILTER_MIN_BUDGET = 3
_FILTER_MAX_BUDGET = 50
_FILTER_FACTOR = 3
_ASSIMILATION_MIN_BUDGET = 10
_ASSIMILATION_MAX_BUDGET = 75
_ASSIMILATION_FACTOR = 5
_FACTS_NOTICE = "\n... [facts 已截断，可调用 read_facts 读取完整版本] ...\n"
_ALLOWED_SERENA_TOOLS = frozenset(
    {"search_for_pattern", "find_file", "get_symbols_overview", "find_symbol", "find_referencing_symbols", "read_file"}
)
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
            f"  - scope: `{insight.scope}`",
            f"  - confidence: `{insight.confidence}`",
            f"  - content: {_truncate_block(insight.content, max_chars=800)}",
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


class _DigestStageAgent(BaseReActAgent):
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
        allow_write_annotation: bool,
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
        self._allow_write_annotation = allow_write_annotation
        self._allow_read_annotation = allow_read_annotation
        self._annotation_paths = _iter_available_annotation_paths(self._project_root)

    def _budget_note(self) -> str:
        remaining = max(0, self._tool_budget_total - self._tool_budget_used)
        return (
            f"\n\n[预算提示] 已使用 {self._tool_budget_used}/{self._tool_budget_total} 次工具预算，"
            f"剩余 {remaining} 次。请优先高价值、窄范围探索。"
        )

    async def on_before_round(self, state: LoopState) -> None:
        self._sync_budget_from_shared()
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
                if name in _ALLOWED_SERENA_TOOLS:
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
            return {
                "tool_call_id": _get_tool_call_id(tool_call),
                "name": tool_name or "unknown",
                "content": blocked_reason,
            }
        result = await super()._invoke_serena(tool_call, serena_client, state=state)
        if tool_name in _ALLOWED_SERENA_TOOLS and self._tool_budget_used < self._tool_budget_total:
            self._consume_budget()
            result["content"] = str(result.get("content") or "") + self._budget_note()
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
            "- stageA/Stage2 默认不读代码；若允许读取 diff，只能通过受控 diff 工具访问。",
            "- stageB 可在预算内做少量代码探索，但 annotation 仍只能通过受控工具访问。",
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
        if self._allow_write_annotation:
            tools.append({
                "type": "function",
                "function": {
                    "name": "write_annotation",
                    "description": "直接写入指定 annotation 文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "markdown": {"type": "string"},
                        },
                        "required": ["path", "markdown"],
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
            return _ok(self._facts_text.replace(_FACTS_NOTICE, ""))
        if name == "read_annotation":
            if not self._allow_read_annotation:
                return _err("当前阶段不允许读取 annotation")
            path = str(args.get("path") or "").strip()
            if not _annotation_path_allowed(path):
                return _err("annotation path 不合法")
            self._consume_budget()
            return _ok(await _read_text_if_exists(self._project_root / _normalize_rel_path(path)))
        if name == "write_annotation":
            path = str(args.get("path") or "").strip()
            markdown = str(args.get("markdown") or "").strip()
            if not self._allow_write_annotation:
                return _err("当前阶段不允许写 annotation")
            if not _annotation_path_allowed(path):
                return _err("annotation path 不合法")
            if not markdown:
                return _err("markdown 不能为空")
            await _write_text_atomic(
                self._project_root / _normalize_rel_path(path),
                markdown.rstrip() + "\n",
            )
            if path not in self._annotation_paths:
                self._annotation_paths.append(path)
            return _ok(f"annotation 已写入: {path}")
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


class DigestFilterStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        insights: list[InsightFact],
        facts_text: str,
        facts_truncated: bool,
        shared_budget: SharedToolBudget | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=_compute_filter_budget(len(insights)),
            shared_budget=shared_budget,
            allow_serena=False,
            allow_write_annotation=False,
            model=model,
            provider=provider,
        )
        self._decision: FilterDecision | None = None

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> FilterDecision:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stageA。请筛选哪些 insight 值得进入下一阶段。"
                "若需要核对是否与现有认知重复，可读取 annotation。不要读代码，不要判断写入目标层级。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._decision is None:
            raise RuntimeError("stageA 未提交筛选结果")
        return self._decision

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestFilterAgent，负责轻量筛选哪些 insight 值得进入认知沉淀阶段。",
                "",
                "## 目标",
                "- 保留真正有新增认知价值的 insight。",
                "- 过滤低价值、明显重复、或明显不值得沉淀的 insight。",
                "",
                "## 约束",
                "- 本阶段不读代码，只允许读取 annotation。",
                "- 本阶段不判断应写入哪个认知文件，也不做写入操作。",
                "- 默认优先依赖直接注入的 insights；只有在需要判断是否与现有认知重复时才读取 annotation。",
                "- 工具预算有限，应优先高价值核对。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若当前 injected insights 已足够，请直接调用 `deliver_stage_result` 提交筛选结果；"
            "若确需核对重复性，只能读取 annotation。不要自然语言总结。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return "stageA 工具预算已耗尽。请停止探索，直接提交当前最保守可用的筛选结果。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._decision is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交筛选结果，再 deliver。",
            })
        return DeliverDecision.finish(
            _safe_json_dumps(
                {
                    "keep_insight_ids": self._decision.keep_insight_ids,
                    "drop_insight_ids": self._decision.drop_insight_ids,
                    "notes": self._decision.notes,
                }
            )
        )

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交 stageA 的筛选结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keep_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "drop_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "notes": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["keep_insight_ids", "drop_insight_ids"],
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
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        keep_ids = args.get("keep_insight_ids")
        drop_ids = args.get("drop_insight_ids")
        notes = args.get("notes") or []
        if not isinstance(keep_ids, list) or not all(isinstance(item, str) for item in keep_ids):
            return err_builder("keep_insight_ids 必须是字符串数组")
        if not isinstance(drop_ids, list) or not all(isinstance(item, str) for item in drop_ids):
            return err_builder("drop_insight_ids 必须是字符串数组")
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            return err_builder("notes 必须是字符串数组")
        self._decision = FilterDecision(
            keep_insight_ids=list(dict.fromkeys(item.strip() for item in keep_ids if item.strip())),
            drop_insight_ids=list(dict.fromkeys(item.strip() for item in drop_ids if item.strip())),
            notes=[item.strip() for item in notes if item.strip()],
        )
        return ok_builder("stageA 结果已接收")


def _render_dirty_file(item: str) -> str:
    return f"- `{item}`"


def build_stale_check_facts_text(
    *,
    insights: list[InsightFact],
    dirty_files: list[str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是 digest stage2 直接注入的 facts。你的任务是判断 insight 是否因 dirty files 存在陈旧风险。",
        "",
        "## Insights",
    ]
    lines.extend(_render_insight(item) for item in insights)
    lines.extend(["", "## Dirty Files"])
    lines.extend([_render_dirty_file(item) for item in dirty_files] or ["- (none)"])
    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=32000,
        notice=_FACTS_NOTICE,
        min_chars=4000,
    )
    return fitted, fitted != full


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


class DigestStaleCheckStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        insights: list[InsightFact],
        facts_text: str,
        facts_truncated: bool,
        dirty_files: list[str],
        shared_budget: SharedToolBudget | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=_compute_filter_budget(len(insights)),
            shared_budget=shared_budget,
            allow_serena=False,
            allow_write_annotation=False,
            allow_read_annotation=False,
            model=model,
            provider=provider,
        )
        self._decision: FilterDecision | None = None
        self._dirty_files = {str(item).strip() for item in dirty_files if str(item).strip()}

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> FilterDecision:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stage2。请先判断哪些 insight 与哪些 dirty files 存在显著关联，"
                "再按需读取少量 baseline diff，判断 insight 是否存在陈旧性风险。"
                "最后提交 keep/drop 结果。不要读取 annotation，不要进入写入层级判断。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._decision is None:
            raise RuntimeError("stage2 未提交筛选结果")
        return self._decision

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestStaleCheckAgent，负责判断 insight 是否因 dirty files 存在陈旧风险。",
                "",
                "## 目标",
                "- 先做 insight 与 dirty files 的显著关联绑定，不要强行绑定。",
                "- 只对显著相关的 dirty files 读取 baseline diff。",
                "- 若 insight 已明显陈旧且不再值得进入内化，直接 drop。",
                "- 若 insight 仍有价值，即使存在一定陈旧风险，也可 keep 交给 stageB。",
                "",
                "## 约束",
                "- 本阶段不读 annotation，不读代码全文，只允许读取受控 baseline diff。",
                "- 工具预算与 stage1 共享，应优先最可能相关的 dirty files。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若路径级信息已足够，请直接提交结果；"
            "若确需判断陈旧风险，只能读取少量 dirty files 的 baseline diff。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return "stage2 工具预算已耗尽。请停止探索，直接提交当前最保守可用的筛选结果。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._decision is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交 stage2 筛选结果，再 deliver。",
            })
        return DeliverDecision.finish(
            _safe_json_dumps(
                {
                    "keep_insight_ids": self._decision.keep_insight_ids,
                    "drop_insight_ids": self._decision.drop_insight_ids,
                    "notes": self._decision.notes,
                }
            )
        )

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file_diff",
                    "description": "读取 dirty files 相对 PCE baseline 的 diff 摘要。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["paths"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交 stage2 的筛选结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keep_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "drop_insight_ids": {"type": "array", "items": {"type": "string"}},
                            "notes": {"type": "array", "items": {"type": "string"}},
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
        if name == "read_file_diff":
            raw_paths = args.get("paths")
            if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
                return err_builder("paths 必须是字符串数组")
            normalized = []
            for item in raw_paths[:10]:
                path = _normalize_rel_path(item)
                if path and path in self._dirty_files and path not in normalized:
                    normalized.append(path)
            if not normalized:
                return err_builder("没有可读取的 dirty file diff")
            self._consume_budget()
            chunks: list[str] = []
            for path in normalized:
                baseline = await load_file_baseline(path, root_path=self._project_root)
                old_content = baseline.content if baseline is not None else None
                abs_path = self._project_root / path
                if abs_path.exists():
                    new_content = await _read_text_if_exists(abs_path)
                else:
                    new_content = None
                chunks.append(
                    _build_file_diff_excerpt(
                        path=path,
                        old_content=old_content,
                        new_content=new_content,
                        max_lines=50,
                    )
                )
            return ok_builder("\n\n".join(chunks) + self._budget_note())
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        keep_ids = args.get("keep_insight_ids")
        drop_ids = args.get("drop_insight_ids")
        notes = args.get("notes") or []
        if not isinstance(keep_ids, list) or not all(isinstance(item, str) for item in keep_ids):
            return err_builder("keep_insight_ids 必须是字符串数组")
        if not isinstance(drop_ids, list) or not all(isinstance(item, str) for item in drop_ids):
            return err_builder("drop_insight_ids 必须是字符串数组")
        if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
            return err_builder("notes 必须是字符串数组")
        self._decision = FilterDecision(
            keep_insight_ids=list(dict.fromkeys(item.strip() for item in keep_ids if item.strip())),
            drop_insight_ids=list(dict.fromkeys(item.strip() for item in drop_ids if item.strip())),
            notes=[item.strip() for item in notes if item.strip()],
        )
        return ok_builder("stage2 结果已接收")


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
            allow_serena=True,
            allow_write_annotation=True,
            model=model,
            provider=provider,
        )
        self._result: AssimilationResult | None = None

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> AssimilationResult:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stageB。请基于保留下来的 insights、dirty files 和 patch evidence，"
                "在有限预算下尽量完成认知内化。你可以读取/写入 annotation，必要时做少量高价值代码探索。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
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
                "- 若核验后认为无需改动，也可以直接结束本轮。",
                "",
                "## 约束",
                "- 优先依赖直接注入的 insights、dirty files、patch evidence。",
                "- 先基于已注入 facts 判断；若仍不足，优先读取 `Readable Annotations` 中的 annotation。",
                "- 只有 annotation 与已注入 facts 仍不足时，才做少量高价值、窄范围代码探索。",
                "- 由你自行判断应修改哪些 annotation 文件；Python 不会预先决定层级。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若当前 facts 已足够，请直接写入 annotation（如需要）并调用 `deliver_stage_result` 收口；"
            "若仍需补证据，再做少量高价值探索。不要自然语言总结。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return "stageB 工具预算已耗尽。请停止探索，基于当前证据直接收口。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._result is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交极简结果，再 deliver。",
            })
        return DeliverDecision.finish(_safe_json_dumps({"summary": self._result.summary}))

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
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
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        self._result = AssimilationResult(summary=str(args.get("summary") or "").strip())
        return ok_builder("stageB 结果已接收")


class DigestCleanupStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
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
            tool_budget=max(0, len(dirty_files) * 3),
            shared_budget=None,
            allow_serena=False,
            allow_write_annotation=False,
            allow_read_annotation=True,
            model=model,
            provider=provider,
        )
        self._result: CleanupResult | None = None
        self._dirty_files = {str(item).strip() for item in dirty_files if str(item).strip()}

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> CleanupResult:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前是 digest stageC。请判断已有 annotation 是否因 dirty files 存在陈旧性风险。"
                "你可以读取 annotation，必要时读取完整 diff。若确认存在陈旧风险，只做局部清理："
                "要么局部重置为骨架，要么按 `##` 标题重写对应 section。无需修改时可直接结束。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._result is None:
            raise RuntimeError("stageC 未提交结果")
        return self._result

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestCleanupAgent，负责清理因 dirty files 而可能陈旧的 annotation。",
                "",
                "## 目标",
                "- 识别受 dirty files 影响而可能误导的现有 annotation。",
                "- 只做清理，不做补正、不做扩写。",
                "- 清理动作仅限：局部重置为骨架，或按 `##` 标题重写 section。",
                "",
                "## 约束",
                "- 优先使用已注入的 dirty file diff 摘要与 annotation 列表判断。",
                "- 需要更多证据时，可读取完整 diff 或读取 annotation。",
                "- 允许 noop；若未发现陈旧风险，不要为了完成任务而修改。",
                "- 最终必须先调用 `deliver_stage_result`，再调用 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return "若当前 facts 已足够且无需清理，请直接提交结果；若需判断陈旧性，只读少量 annotation 或完整 diff。"

    def _build_budget_exhausted_prompt(self) -> str:
        return "stageC 工具预算已耗尽。请停止探索，基于当前证据直接收口。"

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._result is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交 stageC 结果，再 deliver。",
            })
        return DeliverDecision.finish(_safe_json_dumps({"summary": self._result.summary}))

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
                    "description": "按 `##` 标题重写 annotation 中的 section 正文。",
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
                    "description": "提交 stageC 的极简完成说明。",
                    "parameters": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
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
            raw_paths = args.get("paths")
            if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
                return err_builder("paths 必须是字符串数组")
            normalized = []
            for item in raw_paths[:10]:
                path = _normalize_rel_path(item)
                if path and path in self._dirty_files and path not in normalized:
                    normalized.append(path)
            if not normalized:
                return err_builder("没有可读取的 dirty file diff")
            self._consume_budget()
            chunks: list[str] = []
            for path in normalized:
                baseline = await load_file_baseline(path, root_path=self._project_root)
                old_content = baseline.content if baseline is not None else None
                abs_path = self._project_root / path
                new_content = await _read_text_if_exists(abs_path) if abs_path.exists() else None
                chunks.append(_build_file_diff_excerpt(path=path, old_content=old_content, new_content=new_content, max_lines=5000))
            return ok_builder("\n\n".join(chunks) + self._budget_note())
        if name == "rewrite_sections":
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
            existing = await _read_text_if_exists(self._project_root / _normalize_rel_path(path))
            if not existing:
                return err_builder("annotation 不存在")
            header_lines, sections, order = _split_markdown_sections(existing)
            for heading, content in section_map.items():
                if heading not in sections:
                    continue
                body = [line.rstrip() for line in content.splitlines()]
                while body and not body[0].strip():
                    body.pop(0)
                while body and not body[-1].strip():
                    body.pop()
                sections[heading] = body
            await _write_text_atomic(
                self._project_root / _normalize_rel_path(path),
                _render_markdown_sections(header_lines, sections, order),
            )
            self._consume_budget()
            return ok_builder(f"sections 已重写: {path}" + self._budget_note())
        if name == "reset_annotation_to_skeleton":
            path = str(args.get("path") or "").strip()
            if not _annotation_path_allowed(path):
                return err_builder("annotation path 不合法")
            skeleton = await _build_annotation_skeleton(self._project_root, path)
            await _write_text_atomic(self._project_root / _normalize_rel_path(path), skeleton)
            self._consume_budget()
            return ok_builder(f"annotation 已重置为骨架: {path}" + self._budget_note())
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        self._result = CleanupResult(summary=str(args.get("summary") or "").strip())
        return ok_builder("stageC 结果已接收")


async def run_digest_filter(
    *,
    project_root: Path,
    insights: list[InsightFact],
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
    shared_budget: SharedToolBudget | None = None,
) -> FilterDecision:
    annotation_paths = _iter_available_annotation_paths(project_root)
    facts_text, truncated = build_filter_facts_text(
        insights=insights,
        annotation_paths=annotation_paths,
        model=model or "gpt-4o-mini",
    )
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        agent = DigestFilterStageAgent(
            project_root=project_root,
            insights=insights,
            facts_text=facts_text,
            facts_truncated=truncated,
            shared_budget=shared_budget,
            model=model,
            provider=provider,
        )
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= 3:
                break
            feedback = (
                f"上一次 stageA 尝试失败（第 {attempt}/3 次）：{exc}\n"
                "请修正工具调用或结果格式；若当前信息已足够，应直接提交筛选结果。"
            )
    assert last_exc is not None
    raise last_exc


async def run_digest_stale_check(
    *,
    project_root: Path,
    insights: list[InsightFact],
    dirty_files: list[str],
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
    shared_budget: SharedToolBudget | None = None,
) -> FilterDecision:
    facts_text, truncated = build_stale_check_facts_text(
        insights=insights,
        dirty_files=dirty_files,
        model=model or "gpt-4o-mini",
    )
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        agent = DigestStaleCheckStageAgent(
            project_root=project_root,
            insights=insights,
            facts_text=facts_text,
            facts_truncated=truncated,
            dirty_files=dirty_files,
            shared_budget=shared_budget,
            model=model,
            provider=provider,
        )
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= 3:
                break
            feedback = (
                f"上一次 stage2 尝试失败（第 {attempt}/3 次）：{exc}\n"
                "请修正工具调用或结果格式；若路径级事实已足够，应直接提交筛选结果。"
            )
    assert last_exc is not None
    raise last_exc


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
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        agent = DigestAssimilationStageAgent(
            project_root=project_root,
            insight_count=len(insights),
            facts_text=facts_text,
            facts_truncated=truncated,
            model=model,
            provider=provider,
        )
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= 3:
                break
            feedback = (
                f"上一次 stageB 尝试失败（第 {attempt}/3 次）：{exc}\n"
                "请修正工具调用；若当前证据已足够，应直接完成内化并提交结果。"
            )
    assert last_exc is not None
    raise last_exc


def build_cleanup_facts_text(
    *,
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    annotation_paths: list[str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是 digest stageC 直接注入的 facts。目标是识别并清理可能陈旧的 annotation。",
        "",
        "## Dirty Files",
    ]
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


async def run_digest_cleanup(
    *,
    project_root: Path,
    dirty_files: list[str],
    patch_facts: list[ChangedFileFact],
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
) -> CleanupResult:
    annotation_paths = _iter_available_annotation_paths(project_root)
    facts_text, truncated = build_cleanup_facts_text(
        dirty_files=dirty_files,
        patch_facts=patch_facts,
        annotation_paths=annotation_paths,
        model=model or "gpt-4o-mini",
    )
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        agent = DigestCleanupStageAgent(
            project_root=project_root,
            dirty_files=dirty_files,
            facts_text=facts_text,
            facts_truncated=truncated,
            model=model,
            provider=provider,
        )
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= 3:
                break
            feedback = (
                f"上一次 stageC 尝试失败（第 {attempt}/3 次）：{exc}\n"
                "请修正工具调用；若无需清理，也应直接提交结果。"
            )
    assert last_exc is not None
    raise last_exc
