"""Digest Router — 轻量 digest 路由与计划阶段。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotation_writer import refresh_navigation_from_snapshot
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
from .digest_delta_builder import DigestDeltaBuilder
from .insight_cache import InsightCache
from .memory import NAVIGATION_TREE_FILE
from .models import ModuleDigestDelta, NavigationTree
from .module_registry import ModuleRegistryManager
from .serena_client import SerenaClient
from .staging import DirtyState

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SECONDS = 180.0
_ROUTER_TOOL_BUDGET = 5
_ROUTER_MAX_ATTEMPTS = 3
_ALLOWED_SERENA_TOOL_NAMES = frozenset(
    {"search_for_pattern", "find_file", "get_symbols_overview", "find_symbol", "read_file"}
)
_ROUTER_FAILURE_MESSAGES: dict[str, str] = {
    REACT_TIMEOUT_BUDGET: "DigestRouterAgent ReAct 循环超时，已达最大时间预算",
    REACT_NO_TOOL_EXHAUSTED: "DigestRouterAgent 连续未产生 tool_calls，已终止",
    REACT_LENGTH_EXHAUSTED: "DigestRouterAgent 输出连续被截断，已放弃本轮",
    REACT_TIMEOUT: "DigestRouterAgent 模型调用超时，重试次数耗尽",
    REACT_LLM_EXHAUSTED: "DigestRouterAgent 模型降级链耗尽，已终止",
}


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


async def _load_navigation_tree(root: Path) -> NavigationTree | None:
    path = _navigation_tree_path(root)
    try:
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        return NavigationTree.model_validate(json.loads(raw))
    except Exception:
        return None


def _build_area_maps(
    tree: NavigationTree | None,
) -> tuple[dict[str, str], dict[str, str]]:
    module_to_area: dict[str, str] = {}
    area_names: dict[str, str] = {}
    if tree is None:
        return module_to_area, area_names
    for area in tree.areas:
        area_names[area.slug] = area.display_name
        for slug in area.module_slugs:
            module_to_area[slug] = area.slug
    return module_to_area, area_names


def _classify_task_kind(deltas: list[ModuleDigestDelta]) -> str:
    has_dirty = any(delta.changed_files for delta in deltas)
    has_insight = any(delta.related_insights for delta in deltas)
    if has_dirty and has_insight:
        return "mixed"
    if has_dirty:
        return "dirty_only"
    return "insight_only"


def _collect_dirty_paths(deltas: list[ModuleDigestDelta]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for delta in deltas:
        for file_fact in delta.changed_files:
            path = str(file_fact.path)
            if path not in seen:
                seen.add(path)
                result.append(path)
    return result


def _collect_insight_ids(deltas: list[ModuleDigestDelta]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for delta in deltas:
        for insight in delta.related_insights:
            if insight.id not in seen:
                seen.add(insight.id)
                result.append(insight.id)
    return result


@dataclass
class DigestTaskV2:
    id: str
    task_level: str
    task_kind: str
    affected_module_slugs: list[str]
    affected_area_slugs: list[str]
    dirty_files: list[str]
    related_insight_ids: list[str]
    deltas: list[ModuleDigestDelta]
    warnings: list[str] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_level": self.task_level,
            "task_kind": self.task_kind,
            "affected_module_slugs": list(self.affected_module_slugs),
            "affected_area_slugs": list(self.affected_area_slugs),
            "dirty_files_count": len(self.dirty_files),
            "related_insights_count": len(self.related_insight_ids),
            "warnings": list(self.warnings),
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_level": self.task_level,
            "task_kind": self.task_kind,
            "affected_module_slugs": list(self.affected_module_slugs),
            "affected_area_slugs": list(self.affected_area_slugs),
            "dirty_files": list(self.dirty_files),
            "related_insight_ids": list(self.related_insight_ids),
            "warnings": list(self.warnings),
            "module_deltas": [delta.model_dump(mode="json") for delta in self.deltas],
        }


@dataclass
class ModuleUpdatePlan:
    module_slug: str
    action: str
    rationale: str = ""

    @staticmethod
    def from_payload(payload: Any) -> ModuleUpdatePlan | None:
        if not isinstance(payload, dict):
            return None
        slug = str(payload.get("module_slug") or "").strip()
        action = str(payload.get("action") or "").strip()
        if not slug or not action:
            return None
        return ModuleUpdatePlan(
            module_slug=slug,
            action=action,
            rationale=str(payload.get("rationale") or "").strip(),
        )


@dataclass
class DigestRouteDecision:
    decision: str
    change_class: str
    confidence: str
    rationale: str
    requires_navigation_update: bool
    module_update_plans: list[ModuleUpdatePlan]
    unresolved_reason: str = ""

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> DigestRouteDecision:
        decision = str(payload.get("decision") or "").strip()
        if decision not in {
            "no_update",
            "update_module_only",
            "update_navigation_only",
            "update_navigation_and_modules",
            "unresolved",
        }:
            raise ValueError(f"非法 decision: {decision or '(empty)'}")
        module_update_plans = [
            plan
            for plan in (
                ModuleUpdatePlan.from_payload(item)
                for item in (payload.get("module_update_plans") or [])
            )
            if plan is not None
        ]
        return DigestRouteDecision(
            decision=decision,
            change_class=str(payload.get("change_class") or "unknown").strip() or "unknown",
            confidence=str(payload.get("confidence") or "medium").strip() or "medium",
            rationale=str(payload.get("rationale") or "").strip(),
            requires_navigation_update=bool(payload.get("requires_navigation_update")),
            module_update_plans=module_update_plans,
            unresolved_reason=str(payload.get("unresolved_reason") or "").strip(),
        )


@dataclass
class DigestPlanResult:
    tasks: list[DigestTaskV2]
    warnings: list[str]


class DigestPlannerV2:
    """将 dirty / insight 归并为更高层级的 digest 任务。"""

    def __init__(self, project_root: Path, insight_cache: InsightCache) -> None:
        self._project_root = project_root.resolve()
        self._insight_cache = insight_cache

    async def build(self, dirty_state: DirtyState) -> DigestPlanResult:
        builder = DigestDeltaBuilder(self._project_root, self._insight_cache)
        module_deltas = await builder.build_for_changes(
            changed_files=dirty_state.changed,
            deleted_files=dirty_state.deleted,
        )

        dirty_paths = set(dirty_state.changed + dirty_state.deleted)
        covered_paths = {
            str(file_fact.path)
            for delta in module_deltas
            for file_fact in delta.changed_files
        }
        unresolved_dirty = sorted(dirty_paths - covered_paths)
        warnings = [
            f"dirty 文件暂未命中 module registry，后续需补归属: {path}"
            for path in unresolved_dirty
        ]

        if not module_deltas:
            tasks: list[DigestTaskV2] = []
            if unresolved_dirty:
                tasks.append(
                    DigestTaskV2(
                        id="task:unresolved-dirty",
                        task_level="unresolved",
                        task_kind="dirty_only",
                        affected_module_slugs=[],
                        affected_area_slugs=[],
                        dirty_files=unresolved_dirty,
                        related_insight_ids=[],
                        deltas=[],
                        warnings=list(warnings),
                    )
                )
            return DigestPlanResult(tasks=tasks, warnings=warnings)

        tree = await _load_navigation_tree(self._project_root)
        module_to_area, _ = _build_area_maps(tree)

        grouped: dict[str, list[ModuleDigestDelta]] = defaultdict(list)
        unresolved_group: list[ModuleDigestDelta] = []
        for delta in module_deltas:
            area_slug = module_to_area.get(delta.module_slug)
            if area_slug:
                grouped[area_slug].append(delta)
            else:
                unresolved_group.append(delta)

        tasks: list[DigestTaskV2] = []
        if unresolved_dirty or unresolved_group:
            unresolved_deltas = list(unresolved_group)
            tasks.append(
                self._build_task(
                    task_id="task:unresolved",
                    task_level="unresolved",
                    affected_area_slugs=[],
                    deltas=unresolved_deltas,
                    extra_dirty=unresolved_dirty,
                    extra_warnings=list(warnings),
                )
            )

        if not grouped:
            return DigestPlanResult(tasks=tasks, warnings=warnings)

        if len(grouped) == 1 and not unresolved_dirty and not unresolved_group:
            area_slug, deltas = next(iter(grouped.items()))
            if len(deltas) == 1:
                delta = deltas[0]
                tasks.append(
                    self._build_task(
                        task_id=f"task:module:{delta.module_slug}",
                        task_level="module",
                        affected_area_slugs=[area_slug],
                        deltas=[delta],
                    )
                )
            else:
                tasks.append(
                    self._build_task(
                        task_id=f"task:area:{area_slug}",
                        task_level="area",
                        affected_area_slugs=[area_slug],
                        deltas=deltas,
                    )
                )
            return DigestPlanResult(tasks=tasks, warnings=warnings)

        all_grouped_deltas: list[ModuleDigestDelta] = []
        affected_areas: list[str] = []
        for area_slug, deltas in grouped.items():
            affected_areas.append(area_slug)
            all_grouped_deltas.extend(deltas)
        tasks.append(
            self._build_task(
                task_id="task:project",
                task_level="project",
                affected_area_slugs=affected_areas,
                deltas=all_grouped_deltas,
            )
        )
        return DigestPlanResult(tasks=tasks, warnings=warnings)

    def _build_task(
        self,
        *,
        task_id: str,
        task_level: str,
        affected_area_slugs: list[str],
        deltas: list[ModuleDigestDelta],
        extra_dirty: list[str] | None = None,
        extra_warnings: list[str] | None = None,
    ) -> DigestTaskV2:
        module_slugs = [delta.module_slug for delta in deltas]
        dirty_files = _collect_dirty_paths(deltas)
        if extra_dirty:
            for path in extra_dirty:
                if path not in dirty_files:
                    dirty_files.append(path)
        return DigestTaskV2(
            id=task_id,
            task_level=task_level,
            task_kind=_classify_task_kind(deltas) if deltas else "dirty_only",
            affected_module_slugs=module_slugs,
            affected_area_slugs=list(affected_area_slugs),
            dirty_files=dirty_files,
            related_insight_ids=_collect_insight_ids(deltas),
            deltas=deltas,
            warnings=list(extra_warnings or []),
        )


class DigestRouterAgent(BaseReActAgent):
    """digest 轻量路由 Agent，只判断需不需要更新以及更新边界。"""

    _completion_temperature = 0.1
    _enable_budget_warning = False

    def __init__(
        self,
        *,
        project_root: Path,
        task: DigestTaskV2,
        model: str | None = None,
        provider: str | None = None,
        model_fallbacks: list[str] | None = None,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
    ) -> None:
        super().__init__(
            model=model,
            provider=provider,
            model_fallbacks=model_fallbacks,
            max_seconds=max_seconds,
        )
        self._project_root = project_root.resolve()
        self._task = task
        self._deliver_tool = DELIVER_TOOL
        self._route_payload: dict[str, Any] | None = None
        self._virtual_tool_names_set: set[str] = set()
        self._tool_budget_total = _ROUTER_TOOL_BUDGET
        self._tool_budget_used = 0
        self._budget_exhausted_notified = False

    async def run(
        self,
        *,
        serena_client: SerenaClient,
        retry_feedback: str | None = None,
    ) -> DigestRouteDecision:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt()},
        ]
        if retry_feedback:
            messages.append({"role": "user", "content": retry_feedback})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _ROUTER_FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._route_payload is None:
            raise RuntimeError("DigestRouter 未产出 route decision")
        return DigestRouteDecision.from_payload(self._route_payload)

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 DigestRouterAgent，负责对 digest 任务做轻量路由，不直接改文档。",
                "",
                "## 核心原则",
                "- 代码事实优先；现有 annotation / README / 设计文档只能参考，不能盖过代码事实。",
                "- 先使用 task facts 判断；只有证据不足时，再用 Serena 做少量补充调查。",
                "- 路由阶段工具预算固定较小，应优先做高价值、窄范围探索，而不是重建全量认知。",
                "- 默认保守：若当前认知仍成立或新增 insight 明显重复，可直接 `no_update`。",
                "- `modules/*.md` 是内容层；`navigation_tree` 是结构真源；`index.md` / `areas/*.md` 是渲染产物。",
                "- 若判断涉及结构变化，应要求 navigation update，不要建议直接手改 `index.md` 或 `areas/*.md`。",
                "- 若只是模块内容增量，优先 `update_module_only`。",
                "- 证据不足时允许 `unresolved`，不要编造更新计划。",
                "- 当预算耗尽后，必须基于已掌握证据直接收口，不得继续要求额外探索。",
                "",
                "## 输出要求",
                "- 你必须先调用 `read_task_facts`，必要时再读取 navigation / annotation / Serena 结果。",
                "- 最终通过 `deliver_route_decision(payload)` 提交结构化结果，再调用 `deliver(answer='route done')`。",
                "- `decision` 只能是：`no_update` / `update_module_only` / `update_navigation_only` / `update_navigation_and_modules` / `unresolved`。",
                "- `module_update_plans[*].action` 只能是：`no_change` / `light_update` / `rewrite` / `create_if_missing`。",
                "- 若 `decision=no_update`，通常 `module_update_plans` 为空。",
                "- 若 `decision=update_navigation_and_modules`，需要给出需要更新的模块列表。",
            ]
        )

    def _build_user_prompt(self) -> str:
        return "\n".join(
            [
                "请对当前 digest 任务做路由判断。",
                f"任务级别：`{self._task.task_level}`；任务性质：`{self._task.task_kind}`。",
                "先读 task facts，再决定是否需要额外读取 annotation 或代码。",
            ]
        )

    def build_tools_schema(self, serena_client: SerenaClient, state: LoopState) -> list[dict[str, Any]]:
        filtered = []
        for schema in serena_client.tools_schema:
            try:
                name = schema["function"]["name"]
            except Exception:
                continue
            if (
                name in _ALLOWED_SERENA_TOOL_NAMES
                and self._tool_budget_used < self._tool_budget_total
            ):
                filtered.append(schema)
        virtual_tools = self._build_virtual_tools()
        self._virtual_tool_names_set = {
            schema["function"]["name"] for schema in virtual_tools
        }
        return filtered + virtual_tools + [self._deliver_tool]

    @property
    def virtual_tool_names(self) -> set[str]:
        return self._virtual_tool_names_set

    def build_no_tool_correction(self, state: LoopState, finish_reason: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                "你刚才没有调用工具。请至少先调用 `read_task_facts`，"
                "完成后用 `deliver_route_decision` 提交结果。"
            ),
        }

    async def on_before_round(self, state: LoopState) -> None:
        if (
            self._tool_budget_used >= self._tool_budget_total
            and not self._budget_exhausted_notified
        ):
            self._append(state, {
                "role": "user",
                "content": (
                    "路由阶段工具预算已耗尽。请停止继续探索，"
                    "直接根据目前 facts、已有 annotation 与已读取代码证据收口，"
                    "输出最保守可用的 route decision。"
                ),
            })
            self._budget_exhausted_notified = True

    async def on_deliver(self, args: dict[str, Any] | None, state: LoopState) -> DeliverDecision:
        if self._route_payload is None:
            return DeliverDecision.continue_with(
                {
                    "role": "user",
                    "content": "你尚未调用 `deliver_route_decision`。请先提交结构化路由结果。",
                }
            )
        return DeliverDecision.finish(_safe_json_dumps(self._route_payload))

    async def handle_virtual_tool(self, tool_call: Any, state: LoopState) -> dict[str, Any] | None:
        tc_id = _get_tool_call_id(tool_call)
        name = _get_tool_name(tool_call) or "unknown"
        args = _extract_tool_call_args(tool_call) or {}

        def _ok(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        def _err(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        if name == "read_task_facts":
            return _ok(_safe_json_dumps(self._task.to_payload()))
        if name == "read_navigation_tree":
            return _ok(await _read_text_if_exists(_navigation_tree_path(self._project_root)))
        if name == "read_project_annotation":
            return _ok(await _read_text_if_exists(_annotation_index_path(self._project_root)))
        if name == "read_area_annotation":
            area_slug = str(args.get("area_slug") or "").strip()
            if not area_slug:
                return _err("area_slug 不能为空")
            return _ok(await _read_text_if_exists(_annotation_areas_dir(self._project_root) / f"{area_slug}.md"))
        if name == "read_module_annotation":
            module_slug = str(args.get("module_slug") or "").strip()
            if not module_slug:
                return _err("module_slug 不能为空")
            return _ok(await _read_text_if_exists(_annotation_modules_dir(self._project_root) / f"{module_slug}.md"))
        if name == "deliver_route_decision":
            payload = args.get("payload")
            if not isinstance(payload, dict):
                return _err("payload 必须是 object")
            self._route_payload = payload
            return _ok("route decision 已接收")
        return _err(f"未知虚拟工具: {name}")

    def _budget_note(self) -> str:
        remaining = max(0, self._tool_budget_total - self._tool_budget_used)
        return (
            f"\n\n[预算提示] 已使用 {self._tool_budget_used}/{self._tool_budget_total} 次探索预算，"
            f"剩余 {remaining} 次。请优先做高价值、窄范围探索。"
        )

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_task_facts",
                    "description": "读取当前 digest 任务的完整事实包。必须优先使用。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_navigation_tree",
                    "description": "读取当前 navigation_tree.json。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_project_annotation",
                    "description": "读取项目级 index.md。",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_area_annotation",
                    "description": "读取指定 area 文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {"area_slug": {"type": "string"}},
                        "required": ["area_slug"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_module_annotation",
                    "description": "读取指定模块文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {"module_slug": {"type": "string"}},
                        "required": ["module_slug"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "deliver_route_decision",
                    "description": "提交当前 digest 任务的结构化路由结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "object"},
                        },
                        "required": ["payload"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def _invoke_serena(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
        state: LoopState | None = None,
    ) -> dict[str, Any]:
        result = await super()._invoke_serena(tool_call, serena_client, state=state)
        name = _get_tool_name(tool_call) or ""
        if name in _ALLOWED_SERENA_TOOL_NAMES and self._tool_budget_used < self._tool_budget_total:
            self._tool_budget_used += 1
            result["content"] = str(result.get("content") or "") + self._budget_note()
        return result


async def route_digest_task(
    *,
    project_root: Path,
    task: DigestTaskV2,
    serena_client: SerenaClient,
    model: str | None = None,
    provider: str | None = None,
) -> DigestRouteDecision:
    feedback: str | None = None
    last_exc: Exception | None = None
    for attempt in range(1, _ROUTER_MAX_ATTEMPTS + 1):
        agent = DigestRouterAgent(
            project_root=project_root,
            task=task,
            model=model,
            provider=provider,
        )
        try:
            return await agent.run(
                serena_client=serena_client,
                retry_feedback=feedback,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= _ROUTER_MAX_ATTEMPTS:
                break
            feedback = (
                f"上一次 route 尝试失败（第 {attempt}/{_ROUTER_MAX_ATTEMPTS} 次）：{exc}\n"
                "请根据错误反馈修正结构化输出或工具使用方式。"
            )
    assert last_exc is not None
    raise last_exc


async def refresh_digest_navigation(
    *,
    project_root: Path,
    serena_client: SerenaClient,
    model: str | None = None,
) -> dict[str, Any]:
    return await refresh_navigation_from_snapshot(
        root_path=project_root,
        serena_client=serena_client,
        model=model,
    )
