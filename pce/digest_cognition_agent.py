"""Digest 特化认知 Agent。

保留 router + worker 分层，但执行骨架参考 TopologyCognitionAgent：
- facts 默认直接注入
- 仅保留 read_facts 作为超限兜底
- 有限工具预算 + 强制收口
- 结构化阶段结果 + Python 侧重试校验
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from .digest_router import DigestRouteDecision, DigestTaskV2, ModuleUpdatePlan
from .file_discovery import is_visible_to_agent
from .memory import NAVIGATION_TREE_FILE
from .models import ChangedFileFact, ModuleDigestDelta
from .module_annotation_contract import (
    extract_coverage_file_paths,
    validate_module_annotation_markdown,
)
from .prompt_guard import fit_text_to_budget
from .serena_client import SerenaClient

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SECONDS = 180.0
_ROUTER_TOOL_BUDGET = 5
_ROUTER_MAX_ATTEMPTS = 3
_WORKER_MIN_BUDGET = 10
_WORKER_MAX_BUDGET = 60
_WORKER_FACTOR = 3
_WORKER_MAX_ATTEMPTS = 3
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
_FACTS_NOTICE = "\n... [facts 已截断，可调用 read_facts 读取完整版本] ...\n"


def _annotation_modules_dir(root: Path) -> Path:
    return root / ".pce" / "annotations" / "modules"


def _annotation_areas_dir(root: Path) -> Path:
    return root / ".pce" / "annotations" / "areas"


def _annotation_index_path(root: Path) -> Path:
    return root / ".pce" / "annotations" / "index.md"


def _navigation_tree_path(root: Path) -> Path:
    return root / ".pce" / "annotations" / NAVIGATION_TREE_FILE


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


def _annotation_rel_module(slug: str) -> str:
    return f".pce/annotations/modules/{slug}.md"


def _annotation_rel_area(slug: str) -> str:
    return f".pce/annotations/areas/{slug}.md"


def _annotation_rel_index() -> str:
    return ".pce/annotations/index.md"


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
            lines.append(f"@@ -{block.old_start},{(block.old_end or block.old_start) - block.old_start + 1} +{block.new_start},{(block.new_end or block.new_start) - block.new_start + 1} @@")
        if old:
            lines.extend(f"- {line}" for line in old.splitlines())
        if new:
            lines.extend(f"+ {line}" for line in new.splitlines())
        lines.append("```")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)


def _render_changed_file_fact(file_fact: ChangedFileFact) -> str:
    header = f"- `{file_fact.path}` [{file_fact.status}]"
    patch = _format_patch_blocks(file_fact)
    if not patch:
        return header
    return f"{header}\n{patch}"


def _render_insights(delta: ModuleDigestDelta) -> list[str]:
    lines: list[str] = []
    for insight in delta.related_insights:
        lines.append(
            "\n".join(
                [
                    f"- scope: `{insight.scope}`",
                    f"  - confidence: `{insight.confidence}`",
                    f"  - content: {_truncate_block(insight.content, max_chars=600)}",
                ]
            )
        )
    return lines


def _render_task_standalone_insights(task: DigestTaskV2) -> list[str]:
    lines: list[str] = []
    for insight in task.standalone_insights:
        lines.append(
            "\n".join(
                [
                    f"- scope: `{insight.scope}`",
                    f"  - confidence: `{insight.confidence}`",
                    f"  - content: {_truncate_block(insight.content, max_chars=600)}",
                ]
            )
        )
    return lines


def _render_delta_block(delta: ModuleDigestDelta) -> str:
    lines = [
        f"### 模块 `{delta.module_slug}` / {delta.module_name}",
        f"- `change_scope_hint`: `{delta.change_scope_hint}`",
        "",
        "#### 当前 annotation baseline",
        delta.annotation_baseline.strip() or "(empty)",
        "",
        "#### 变化说明",
    ]
    if delta.changed_files:
        for file_fact in delta.changed_files:
            lines.append(_render_changed_file_fact(file_fact))
    else:
        lines.append("- 当前无 dirty file，仅有 insight 待沉淀。")
    lines.extend(["", "#### 相关 insight"])
    insight_lines = _render_insights(delta)
    lines.extend(insight_lines or ["- 当前无相关 insight。"])
    return "\n".join(lines).strip()


def build_router_facts_text(
    *,
    task: DigestTaskV2,
    project_annotation: str,
    area_annotations: dict[str, str],
    module_annotations: dict[str, str],
    navigation_tree_text: str,
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是直接注入的 digest router facts。优先基于这些事实判断哪些 insight 值得沉淀，以及是否存在陈旧风险。",
        "",
        "## 任务概览",
        f"- task_id: `{task.id}`",
        f"- task_level: `{task.task_level}`",
        f"- task_kind: `{task.task_kind}`",
        f"- affected_areas: {', '.join(f'`{item}`' for item in task.affected_area_slugs) or '(none)'}",
        f"- affected_modules: {', '.join(f'`{item}`' for item in task.affected_module_slugs) or '(none)'}",
        f"- dirty_files_count: {len(task.dirty_files)}",
        f"- related_insights_count: {len(task.related_insight_ids)}",
    ]
    if task.warnings:
        lines.extend(["", "## warnings", *[f"- {item}" for item in task.warnings]])
    if task.dirty_files:
        lines.extend(["", "## dirty files", *[f"- `{item}`" for item in task.dirty_files]])
    for delta in task.deltas:
        lines.extend(["", _render_delta_block(delta)])
    standalone_insights = _render_task_standalone_insights(task)
    if standalone_insights:
        lines.extend(["", "## unresolved insights", *standalone_insights])

    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=12000,
        notice=_FACTS_NOTICE,
        min_chars=2400,
    )
    return fitted, fitted != full


def build_worker_facts_text(
    *,
    task: DigestTaskV2,
    delta: ModuleDigestDelta,
    plan: ModuleUpdatePlan,
    navigation_tree_text: str,
    project_annotation: str,
    area_annotations: dict[str, str],
    model: str,
) -> tuple[str, bool]:
    lines = [
        "以下是直接注入的 digest worker facts。目标是基于 insight 做增量认知沉淀，而不是重建全量认知。",
        "",
        "## 任务概览",
        f"- task_id: `{task.id}`",
        f"- task_level: `{task.task_level}`",
        f"- task_kind: `{task.task_kind}`",
        f"- target_module: `{delta.module_slug}` / {delta.module_name}",
        f"- worker_action: `{plan.action}`",
        f"- rationale: {plan.rationale or '(none)'}",
    ]
    if task.dirty_files:
        lines.extend(["", "## dirty files", *[f"- `{item}`" for item in task.dirty_files]])
    lines.extend(["", _render_delta_block(delta)])

    full = "\n".join(lines).strip()
    fitted = fit_text_to_budget(
        model,
        full,
        token_budget=14000,
        notice=_FACTS_NOTICE,
        min_chars=2600,
    )
    return fitted, fitted != full


def _compute_worker_budget(delta: ModuleDigestDelta) -> int:
    raw = _WORKER_FACTOR * (len(delta.related_insights) + len(delta.changed_files))
    return max(_WORKER_MIN_BUDGET, min(_WORKER_MAX_BUDGET, raw))


@dataclass
class WorkerResult:
    status: str
    note: str
    module_slug: str


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
        model: str | None = None,
        provider: str | None = None,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
    ) -> None:
        super().__init__(model=model, provider=provider, max_seconds=max_seconds)
        self._project_root = project_root.resolve()
        self._facts_text = facts_text.strip()
        self._facts_truncated = facts_truncated
        self._tool_budget_total = tool_budget
        self._tool_budget_used = 0
        self._budget_exhausted_notified = False
        self._deliver_tool = DELIVER_TOOL
        self._virtual_tool_names_set: set[str] = set()
        self._injected_context_paths: set[str] = set()
        self._readable_module_contexts: dict[str, str] = {}
        self._readable_area_contexts: dict[str, str] = {}
        self._project_index_context: str = ""

    def _budget_note(self) -> str:
        remaining = max(0, self._tool_budget_total - self._tool_budget_used)
        return (
            f"\n\n[预算提示] 已使用 {self._tool_budget_used}/{self._tool_budget_total} 次工具预算，"
            f"剩余 {remaining} 次。请优先高价值、窄范围探索。"
        )

    async def on_before_round(self, state: LoopState) -> None:
        if (
            self._tool_budget_total > 0
            and self._tool_budget_used >= self._tool_budget_total
            and not self._budget_exhausted_notified
        ):
            self._append(state, {
                "role": "user",
                "content": self._build_budget_exhausted_prompt(),
            })
            self._budget_exhausted_notified = True

    def build_tools_schema(self, serena_client: SerenaClient, state: LoopState) -> list[dict[str, Any]]:
        filtered = []
        for schema in serena_client.tools_schema:
            try:
                name = schema["function"]["name"]
            except Exception:
                continue
            if name in _ALLOWED_SERENA_TOOLS and self._tool_budget_used < self._tool_budget_total:
                filtered.append(schema)
        virtual = self._build_virtual_tools()
        self._virtual_tool_names_set = {schema["function"]["name"] for schema in virtual}
        return filtered + virtual + [self._deliver_tool]

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
            self._tool_budget_used += 1
            result["content"] = str(result.get("content") or "") + self._budget_note()
        return result

    def _blocked_serena_access_reason(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str | None:
        relative_path = tool_args.get("relative_path")
        if not isinstance(relative_path, str):
            return None
        normalized = relative_path.strip()
        if not normalized:
            return None
        if is_visible_to_agent(self._project_root, normalized):
            return None
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

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        tools = []
        if self._facts_truncated:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_facts",
                    "description": "读取未截断的完整 facts。仅在当前注入 facts 不足时使用。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
        tools.extend(self._build_context_read_tools())
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
        if name == "read_project_index":
            return _ok(self._project_index_context)
        if name == "read_area_annotation":
            area_slug = str(args.get("area_slug") or "").strip()
            if not area_slug:
                return _err("area_slug 不能为空")
            if area_slug not in self._readable_area_contexts:
                return _err(f"当前不可读取 area 认知: {area_slug}")
            return _ok(self._readable_area_contexts[area_slug])
        if name == "read_module_annotation":
            module_slug = str(args.get("module_slug") or "").strip()
            if not module_slug:
                return _err("module_slug 不能为空")
            if module_slug not in self._readable_module_contexts:
                return _err(f"当前不可读取模块认知: {module_slug}")
            return _ok(self._readable_module_contexts[module_slug])
        return await self._handle_stage_virtual_tool(name, args, _ok, _err)

    def build_no_tool_correction(self, state: LoopState, finish_reason: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": self._build_no_tool_prompt(),
        }

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

    def _build_context_boundary_prompt(self) -> str:
        injected = sorted(self._injected_context_paths)
        readable_modules = sorted(_annotation_rel_module(slug) for slug in self._readable_module_contexts)
        readable_areas = sorted(_annotation_rel_area(slug) for slug in self._readable_area_contexts)
        readable_index = [_annotation_rel_index()] if self._project_index_context and _annotation_rel_index() not in self._injected_context_paths else []
        lines = [
            "## 认知读取边界",
            "- 以下认知已直接注入，无需再次读取：",
        ]
        lines.extend([f"  - `{item}`" for item in injected] or ["  - (none)"])
        lines.extend([
            "- 以下认知可通过受控 read 工具再次读取：",
        ])
        readable = [*readable_index, *readable_areas, *readable_modules]
        lines.extend([f"  - `{item}`" for item in readable] or ["  - (none)"])
        lines.extend([
            "- `.pce/annotations/*` 不可通过 Serena / 通用代码读取工具访问。",
            "- 若需要补充认知，只能调用受控的 `read_project_index` / `read_area_annotation` / `read_module_annotation`。",
            "- Serena 工具只用于读取代码事实，不用于读取认知文档。",
        ])
        return "\n".join(lines)

    def _build_context_read_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if self._project_index_context and _annotation_rel_index() not in self._injected_context_paths:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_project_index",
                    "description": "读取项目级 `index.md` 认知（仅当未直接注入时可用）。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
        if self._readable_area_contexts:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_area_annotation",
                    "description": "读取指定 area 认知文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {"area_slug": {"type": "string"}},
                        "required": ["area_slug"],
                        "additionalProperties": False,
                    },
                },
            })
        if self._readable_module_contexts:
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_module_annotation",
                    "description": "读取指定模块认知文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {"module_slug": {"type": "string"}},
                        "required": ["module_slug"],
                        "additionalProperties": False,
                    },
                },
            })
        return tools


class DigestRouterStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        task: DigestTaskV2,
        facts_text: str,
        facts_truncated: bool,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=_ROUTER_TOOL_BUDGET,
            model=model,
            provider=provider,
        )
        self._task = task
        self._stage_payload: dict[str, Any] | None = None

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> DigestRouteDecision:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                "当前阶段是 digest router。请基于已注入 facts 判断："
                "哪些 insight 值得沉淀、哪些可能因 dirty file 而存在陈旧风险、以及哪些 module 需要最小认知修正。"
                "优先使用 facts；仅在明显不足时再做少量探索。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._stage_payload is None:
            raise RuntimeError("router 未提交阶段结果")
        return DigestRouteDecision.from_payload(self._stage_payload)

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestRouterAgent，负责对单个 digest task 做轻量路由，不直接写文档。",
                "",
                "## 目标",
                "- 判断哪些 insight 值得沉淀，哪些可以直接跳过。",
                "- 判断哪些 insight 可能因 dirty file 而存在陈旧风险。",
                "- 若需要模块更新，指出需要更新的 module slug 以及更新强度。",
                "",
                "## 约束",
                "- 代码事实优先，文档只能参考。",
                "- 优先依赖直接注入的 facts；只有在 facts 不足时才调用 Serena。",
                "- 探索预算很小，只做高价值、窄范围探索。",
                "- 目标是 insight 路由和边界判断，不是重建全量认知。",
                "- 低价值、明显重复、或对既有认知没有新增信息的 insight，可直接 `no_update`。",
                "- dirty file 只是辅助判断 insight 是否可能陈旧，不是独立认知沉淀触发源。",
                "- 最终必须先调用 `deliver_stage_result(payload)`，再调用 `deliver(answer='stage done')`。",
                "- `decision` 只能是 `no_update` / `update_module_only` / `unresolved`。",
                "- `module_update_plans[*].action` 只能是 `light_update` / `rewrite` / `create_if_missing` / `no_change`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若当前 facts 已足够，请直接调用 `deliver_stage_result` 提交路由结果；"
            "若仍需补证据，先调用 Serena 工具。不要直接输出自然语言结论。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return (
            "router 阶段工具预算已耗尽。请停止探索，直接基于当前 facts 和已掌握证据输出最保守可用的 route decision。"
        )

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._stage_payload is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `deliver_stage_result`。请先提交结构化路由结果，再 deliver。",
            })
        return DeliverDecision.finish(_safe_json_dumps(self._stage_payload))

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交 router 阶段的结构化结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {"payload": {"type": "object"}},
                        "required": ["payload"],
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
        if name != "deliver_stage_result":
            return err_builder(f"未知虚拟工具: {name}")
        payload = args.get("payload")
        if not isinstance(payload, dict):
            return err_builder("payload 必须是 object")
        self._stage_payload = payload
        return ok_builder("router 阶段结果已接收")


class DigestWorkerStageAgent(_DigestStageAgent):
    def __init__(
        self,
        *,
        project_root: Path,
        task: DigestTaskV2,
        delta: ModuleDigestDelta,
        plan: ModuleUpdatePlan,
        facts_text: str,
        facts_truncated: bool,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            project_root=project_root,
            facts_text=facts_text,
            facts_truncated=facts_truncated,
            tool_budget=_compute_worker_budget(delta),
            model=model,
            provider=provider,
        )
        self._task = task
        self._delta = delta
        self._plan = plan
        self._result: WorkerResult | None = None

    async def run(self, *, serena_client: SerenaClient, retry_feedback: str | None = None) -> WorkerResult:
        messages = self._build_base_messages()
        messages.append({
            "role": "user",
            "content": (
                f"当前阶段是 digest worker，目标模块是 `{self._delta.module_slug}`。"
                "请基于已注入的 insight 与 dirty file 事实，"
                "判断当前 insight 是否可能陈旧，并做最小必要的认知修正。若证据不足，可在预算内做少量探索。"
                "目标是沉淀增量认知，不是重建全量认知。"
            ),
        })
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._result is None:
            raise RuntimeError("worker 未提交模块结果")
        return self._result

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestWorkerAgent，负责对单个模块做增量认知沉淀。",
                "",
                "## 目标",
                "- 对当前模块做最小必要的 annotation 更新。",
                "- 若确认当前模块无需更新，也应明确跳过并说明原因。",
                "",
                "## 约束",
                "- 代码事实优先，文档只能参考。",
                "- 优先依赖直接注入的 insight / dirty file 事实；只有在 facts 不足时才调用 Serena。",
                "- 探索预算有限，应做高价值、窄范围探索。",
                "- 目标是围绕 insight 修正现有认知，不是重写整个项目认知。",
                "- 若当前 insight 低价值、重复、或不足以改变现有认知，可直接 `mark_module_skipped`。",
                "- dirty file 仅用于辅助判断该 insight 是否可能过期，不应把任务扩展成全量变更调查。",
                "- 文档必须保留固定章节：`## 覆盖文件` `## 核心职责` `## 关键流程` `## 外部协作` `## 风险与约束`。",
                "- `## 关键符号` 可选，且只能保留少量稳定锚点。",
                "- 最终必须调用 `write_module_annotation_and_finish` 或 `mark_module_skipped`，再 `deliver(answer='stage done')`。",
            ]
        )

    def _build_no_tool_prompt(self) -> str:
        return (
            "你刚才没有调用工具。若当前证据已足够，请直接调用 `write_module_annotation_and_finish` 或 `mark_module_skipped` 收口；"
            "若仍需补证据，再调用 Serena 工具。不要直接输出自然语言结论。"
        )

    def _build_budget_exhausted_prompt(self) -> str:
        return (
            "worker 阶段工具预算已耗尽。请停止探索，直接基于当前 facts 和已掌握证据收口："
            "能做最小更新就写回，不能确认就 mark skipped。"
        )

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._result is None:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "你尚未调用 `write_module_annotation_and_finish` 或 `mark_module_skipped`。请先提交模块结果。",
            })
        return DeliverDecision.finish(_safe_json_dumps({
            "module_slug": self._result.module_slug,
            "status": self._result.status,
            "note": self._result.note,
        }))

    def _extra_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "write_module_annotation_and_finish",
                    "description": "写入完整模块 annotation，并标记该模块处理完成。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "markdown": {"type": "string"},
                            "note": {"type": "string"},
                        },
                        "required": ["markdown", "note"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_module_skipped",
                    "description": "当前模块暂不更新 annotation，但要明确写明原因。",
                    "parameters": {
                        "type": "object",
                        "properties": {"note": {"type": "string"}},
                        "required": ["note"],
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
        if name == "write_module_annotation_and_finish":
            markdown = str(args.get("markdown") or "").strip()
            note = str(args.get("note") or "").strip()
            if not markdown:
                return err_builder("markdown 不能为空")
            if not note:
                return err_builder("note 不能为空")
            expected_paths = extract_coverage_file_paths(self._delta.annotation_baseline)
            if not expected_paths:
                expected_paths = [str(item.path) for item in self._delta.changed_files]
            errors = validate_module_annotation_markdown(
                markdown,
                expected_file_paths=expected_paths,
                require_core_responsibility=True,
            )
            if errors:
                return err_builder("写入校验失败: " + " | ".join(errors))
            await _write_text_atomic(
                _annotation_modules_dir(self._project_root) / f"{self._delta.module_slug}.md",
                markdown.rstrip() + "\n",
            )
            self._result = WorkerResult(
                status="done",
                note=note,
                module_slug=self._delta.module_slug,
            )
            return ok_builder(f"模块文档已写入并完成: {self._delta.module_slug}")
        if name == "mark_module_skipped":
            note = str(args.get("note") or "").strip()
            if not note:
                return err_builder("note 不能为空")
            self._result = WorkerResult(
                status="skipped",
                note=note,
                module_slug=self._delta.module_slug,
            )
            return ok_builder(f"模块已标记跳过: {self._delta.module_slug}")
        return err_builder(f"未知虚拟工具: {name}")


async def run_digest_router(
    *,
    project_root: Path,
    task: DigestTaskV2,
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
) -> DigestRouteDecision:
    project_annotation = await _read_text_if_exists(_annotation_index_path(project_root))
    area_annotations = {
        slug: await _read_text_if_exists(_annotation_areas_dir(project_root) / f"{slug}.md")
        for slug in task.affected_area_slugs
    }
    module_annotations = {
        slug: await _read_text_if_exists(_annotation_modules_dir(project_root) / f"{slug}.md")
        for slug in task.affected_module_slugs
    }
    navigation_tree_text = await _read_text_if_exists(_navigation_tree_path(project_root))
    facts_text, truncated = build_router_facts_text(
        task=task,
        project_annotation=project_annotation,
        area_annotations=area_annotations,
        module_annotations=module_annotations,
        navigation_tree_text=navigation_tree_text,
        model=model or "gpt-4o-mini",
    )
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, _ROUTER_MAX_ATTEMPTS + 1):
        agent = DigestRouterStageAgent(
            project_root=project_root,
            task=task,
            facts_text=facts_text,
            facts_truncated=truncated,
            model=model,
            provider=provider,
        )
        agent._project_index_context = project_annotation
        agent._readable_area_contexts = {
            slug: content for slug, content in area_annotations.items() if content.strip()
        }
        agent._readable_module_contexts = {
            slug: content for slug, content in module_annotations.items() if content.strip()
        }
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= _ROUTER_MAX_ATTEMPTS:
                break
            feedback = (
                f"上一次 router 尝试失败（第 {attempt}/{_ROUTER_MAX_ATTEMPTS} 次）：{exc}\n"
                "请修正工具调用或结构化结果；若事实已足够，应直接收口，不要自然语言回答。"
            )
    assert last_exc is not None
    raise last_exc


async def run_digest_worker(
    *,
    project_root: Path,
    task: DigestTaskV2,
    delta: ModuleDigestDelta,
    plan: ModuleUpdatePlan,
    model: str | None,
    provider: str | None,
    serena_client: SerenaClient,
) -> WorkerResult:
    project_annotation = await _read_text_if_exists(_annotation_index_path(project_root))
    area_annotations = {
        slug: await _read_text_if_exists(_annotation_areas_dir(project_root) / f"{slug}.md")
        for slug in task.affected_area_slugs
    }
    navigation_tree_text = await _read_text_if_exists(_navigation_tree_path(project_root))
    all_candidate_module_slugs = sorted(set(task.affected_module_slugs))
    extra_module_annotations = {
        slug: await _read_text_if_exists(_annotation_modules_dir(project_root) / f"{slug}.md")
        for slug in all_candidate_module_slugs
        if slug != delta.module_slug
    }
    facts_text, truncated = build_worker_facts_text(
        task=task,
        delta=delta,
        plan=plan,
        navigation_tree_text=navigation_tree_text,
        project_annotation=project_annotation,
        area_annotations=area_annotations,
        model=model or "gpt-4o-mini",
    )
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, _WORKER_MAX_ATTEMPTS + 1):
        agent = DigestWorkerStageAgent(
            project_root=project_root,
            task=task,
            delta=delta,
            plan=plan,
            facts_text=facts_text,
            facts_truncated=truncated,
            model=model,
            provider=provider,
        )
        agent._project_index_context = project_annotation
        agent._readable_area_contexts = {
            slug: content for slug, content in area_annotations.items() if content.strip()
        }
        agent._readable_module_contexts = {
            slug: content for slug, content in extra_module_annotations.items() if content.strip()
        }
        try:
            return await agent.run(serena_client=serena_client, retry_feedback=feedback)
        except Exception as exc:
            last_exc = exc
            if attempt >= _WORKER_MAX_ATTEMPTS:
                break
            feedback = (
                f"上一次 worker 尝试失败（第 {attempt}/{_WORKER_MAX_ATTEMPTS} 次）：{exc}\n"
                "请不要自然语言总结；若证据已足够，直接写回或 skip。若事实不足，再做极少量高价值探索。"
            )
    assert last_exc is not None
    raise last_exc
