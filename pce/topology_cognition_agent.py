"""TopologyCognitionAgent — init 阶段的轻量拓扑认知 Agent。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
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
from .init_cognition_limits import (
    PCEIGNORE_STAGE_TOOL_BUDGET,
    TOPOLOGY_INCREMENTAL_REPAIR_STAGE_TOOL_BUDGET,
    TOPOLOGY_INCREMENTAL_STAGE_TOOL_BUDGET,
    TOPOLOGY_NAVIGATION_STAGE_TOOL_BUDGET,
    TOPOLOGY_REPAIR_STAGE_TOOL_BUDGET,
)
from .module_annotation_contract import validate_module_annotation_markdown
from .serena_client import SerenaClient

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SECONDS = 180.0
_BATCH_READ_MAX_FILES = 10
_BATCH_READ_MAX_LINES = 50
_ALLOWED_SERENA_TOOL_NAMES_BY_STAGE: dict[str, frozenset[str]] = {
    "pceignore": frozenset({"read_file", "search_for_pattern"}),
    "pceignore_refresh": frozenset({"read_file", "search_for_pattern"}),
    "navigation_tree": frozenset({"find_file", "get_symbols_overview", "read_file"}),
    "navigation_repair": frozenset({"find_file"}),
    "navigation_incremental": frozenset({"find_file", "get_symbols_overview", "read_file"}),
    "navigation_incremental_repair": frozenset({"find_file"}),
    "write_area_modules": frozenset({"search_for_pattern", "find_file", "get_symbols_overview", "find_symbol", "find_referencing_symbols", "read_file"}),
}
_REACT_FAILURE_MESSAGES: dict[str, str] = {
    REACT_TIMEOUT_BUDGET: "TopologyCognitionAgent ReAct 循环超时，已达最大时间预算",
    REACT_NO_TOOL_EXHAUSTED: "TopologyCognitionAgent 连续未产生 tool_calls，已终止",
    REACT_LENGTH_EXHAUSTED: "TopologyCognitionAgent 输出连续被截断，已放弃本轮",
    REACT_TIMEOUT: "TopologyCognitionAgent 模型调用超时，重试次数耗尽",
    REACT_LLM_EXHAUSTED: "TopologyCognitionAgent 模型降级链耗尽，已终止",
}


class TopologyCognitionAgent(BaseReActAgent):
    """用于 init 轻量建图与模块认知直写的特化 Agent。"""

    _temperature_env_key = "PCE_TOPOLOGY_TEMPERATURE"
    _completion_temperature = 0.1
    _enable_budget_warning = False

    def __init__(
        self,
        *,
        project_root: Path,
        discovery_facts: dict[str, Any],
        embedded_facts_text: str | None = None,
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
        self._discovery_facts = discovery_facts
        self._embedded_facts_text = (
            embedded_facts_text.strip()
            if embedded_facts_text and embedded_facts_text.strip()
            else self._build_embedded_facts_text()
        )
        self._deliver_tool = DELIVER_TOOL
        self._active_stage: str | None = None
        self._stage_payload: dict[str, Any] | None = None
        self._stage_context: dict[str, Any] = {}
        self._written_module_slugs_by_area: dict[str, set[str]] = {}
        self._virtual_tool_names_set: set[str] = set()
        self._stage_tool_budget_total = 0
        self._stage_tool_budget_used = 0
        self._stage_budget_exhausted_notified = False

    def build_initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "system", "content": self._embedded_facts_text},
        ]

    async def run_stage(
        self,
        *,
        stage: str,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
        user_prompt: str | None = None,
        stage_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._active_stage = stage
        self._stage_payload = None
        self._stage_context = dict(stage_context or {})
        if stage not in {"write_area_modules", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"}:
            self._stage_context = {}
        if stage in {"pceignore", "pceignore_refresh"}:
            self._stage_tool_budget_total = PCEIGNORE_STAGE_TOOL_BUDGET
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False
        elif stage == "navigation_tree":
            self._stage_tool_budget_total = TOPOLOGY_NAVIGATION_STAGE_TOOL_BUDGET
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False
        elif stage == "navigation_repair":
            self._stage_tool_budget_total = TOPOLOGY_REPAIR_STAGE_TOOL_BUDGET
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False
        elif stage == "navigation_incremental":
            self._stage_tool_budget_total = TOPOLOGY_INCREMENTAL_STAGE_TOOL_BUDGET
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False
        elif stage == "navigation_incremental_repair":
            self._stage_tool_budget_total = TOPOLOGY_INCREMENTAL_REPAIR_STAGE_TOOL_BUDGET
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False
        else:
            self._stage_tool_budget_total = 0
            self._stage_tool_budget_used = 0
            self._stage_budget_exhausted_notified = False

        prompt = user_prompt or self._build_stage_prompt(stage)
        messages.append({"role": "user", "content": prompt})
        summary, _ = await self.run_loop(messages, serena_client)
        failure = _REACT_FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)
        if self._stage_payload is None:
            raise RuntimeError(f"阶段 {stage} 未产出阶段完成标记")
        return self._stage_payload

    def _build_embedded_facts_text(self) -> str:
        discovery_text = json.dumps(self._discovery_facts, ensure_ascii=False, indent=2)
        return "\n".join(
            [
                "以下是直接注入的 discovery facts。优先基于这些事实工作；只有信息不足时再探索代码。",
                "",
                "## discovery_facts",
                discovery_text,
            ]
        )

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "你是 PCE 的 TopologyCognitionAgent，负责在 init 阶段建立轻量拓扑认知。",
                "",
                "## 核心原则",
                "- 唯一稳定事实来源是当前代码结构与代码内容；文档只能参考，不能直接当成事实。",
                "- 优先使用直接注入的 discovery facts；仅在 facts 不足时，再调用 Serena 工具补证据。",
                "- 若需读取代码，先用 Serena 的搜索/符号工具定界，再按需 read_file，避免无界读取。",
                "- 若发现明显扁平的目录结构，可优先用受控批量读取快速查看多个同级文件的局部内容，再判断是否需要拆成单文件模块。",
                "- 目标是形成稳定导航与轻量模块认知，不展开内部实现细节，不穷举符号。",
                "- 若证据不足，宁可保持较粗但稳定的划分，也不要切出很多脆弱小模块。",
                "- stage1 为 `pceignore` 阶段：允许有限探索后给出保守过滤规则，用于减少后续索引噪声。",
                "- `pceignore_refresh` 阶段沿用相同的轻量探索与预算机制，但目标是判断当前 ignore 边界是否需要增量修正。",
                "- 每次工具调用后都会收到剩余预算提示；预算耗尽后必须基于现有证据强制收口。",
                "- 第一阶段若未通过后处理校验，会收到错误反馈要求重试；请根据反馈修正，而不是退回粗糙 fallback。",
                "- 第二阶段按 area 顺序写模块文档；允许多轮修正，直到 Python 侧确认当前 area 完成。",
                "",
                "## 阶段输出约束",
                "- stage1 输出 `pceignore`：提交 `{ignore_patterns:[...]}`。",
                "- `pceignore_refresh` 输出 `{action, ignore_patterns, rationale}`，其中 `action` 只能是 `no_update` 或 `append_patterns`。",
                "- stage2 输出 `navigation_tree`：直接给出 project -> areas -> modules 的完整导航树。",
                "- `navigation_repair` 阶段也输出完整 `navigation_tree`，但目标是基于现有结构补齐遗漏与修正错误挂载。",
                "- `navigation_incremental` 阶段用于 dirty file 触发后的增量导航判定，可输出 `no_change`、`module_update`、`area_rebuild` 或 `full_rebuild`。",
                "- `navigation_incremental_repair` 阶段也输出增量导航判定，但必须基于现有 tree 做更保守、更小范围的修正。",
                "- stage3 不再输出大 JSON；而是直接调用 `write_module_annotation` 写当前 area 下的模块文档。",
                "- 当当前 area 全部处理完后，调用 `mark_area_done`，再调用 `deliver(answer='stage done')`。",
                "",
                "## pceignore payload 要求",
                "- 字段：`ignore_patterns`。",
                "- `ignore_patterns` 必须是字符串数组，元素为 gitignore 风格路径/模式。",
                "- 要保守；宁少勿错，不要误伤可能属于源码、配置、脚本、测试、关键文档的路径。",
                "",
                "## navigation_tree payload 要求",
                "- 字段：`project_summary`, `fallback_area_slug`, `areas`。",
                "- `project_summary` 必须是一段简短概括，说明项目做什么、面向什么问题或场景。",
                "- `areas[*]` 字段：`slug`, `display_name`, `summary`, `modules`, `recommended_order`, `source_prefixes`, `is_fallback`。",
                "- `areas[*].summary` 必须是一段 area 级概括，说明该 area 主要职责与边界。",
                "- `areas[*].modules[*]` 字段：`slug`, `display_name`, `summary`, `module_type`, `include`, `exclude`。",
                "- `module_type` 只能是 `directory`、`file_centered`、`residual`。",
                "- 优先按目录层级建立 `directory` 模块；单文件 `file_centered` 只在平铺结构且该文件确实像职责中心时使用。",
                "- 每个 area 最多保留 1 个 `residual` 模块；它用于承接该 area 内剩余文件，可不写 `include` / `exclude`。",
                "- `include` / `exclude` 使用 glob 风格路径规则表达显式边界；优先按目录层级覆盖，而不是单文件穷举。",
                "- 若宽泛 `include` 会覆盖到更精确的模块，请通过 `exclude` 消除重叠，而不是让同一文件落入多个模块。",
                "- `areas[*].modules[*].summary` 必须是一句 module 入口介绍，用于快速导航。",
                "- 恰好 1 个 fallback area；每个文件只能归属一个 module；每个 module 只能归属一个 area。",
                "- `include` / `exclude` 只能引用当前注入 `files_tree` 可覆盖的候选路径，不要编造索引外文件。",
                "- 不要单独输出 module_catalog，也不要把目录名机械等同于 area/module。",
                "",
                "## navigation_incremental payload 要求",
                "- 字段：`decision`, `rationale`。",
                "- `decision` 只能是 `no_change`、`module_update`、`area_rebuild`、`full_rebuild`。",
                "- 若 `decision` 不是 `no_change`，必须额外提供完整 `navigation_tree` 字段。",
                "- `rationale` 只需简短说明本次为何属于该级别结构变化。",
                "- 增量阶段只处理挂载与导航，不主动改动模块认知正文。",
                "",
                "## write_module_annotation 约束",
                "- 只能写当前 area 下被明确分配的 module slug。",
                "- 文档必须包含：`## 覆盖文件`、`## 核心职责`、`## 关键流程`、`## 外部协作`、`## 风险与约束`。",
                "- `## 关键符号` 可选；即使填写，也只能保留少量稳定锚点。",
                "- `## 覆盖文件` 必须列出真实文件路径；不要编造不存在的路径。",
                "- `## 核心职责` 至少写 1 行；其它章节如证据不足可留空，但标题必须保留。",
                "- 工具若返回校验错误，请根据错误信息修正文档后重试。",
            ]
        )

    def _build_stage_prompt(self, stage: str) -> str:
        if stage == "pceignore":
            return "\n".join(
                [
                    "当前阶段：`pceignore`。",
                    "请基于已注入的折叠目录树摘要与 `.gitignore` 事实，必要时再做极少量探索，输出保守可用的 `.pce/pceignore` 规则。",
                    "本阶段最多允许有限工具探索；每次调用工具后你会收到预算反馈。",
                    "预算耗尽后必须停止探索，并基于当前证据提交结果。",
                    "先提交 pceignore 阶段结果，再调用 deliver(answer=\"stage done\")。",
                ]
            )
        if stage == "pceignore_refresh":
            return "\n".join(
                [
                    "当前阶段：`pceignore_refresh`。",
                    "请基于已注入的 dirty files、当前 `.pce/pceignore` 与折叠目录树摘要，判断当前 ignore 边界是否需要增量更新。",
                    "若当前规则仍然足够，请输出 `action=no_update` 并直接结束。",
                    "若需要更新，请仅输出最小增量 ignore patterns，避免重建整份 `.pce/pceignore`。",
                    "本阶段允许有限工具探索；每次调用工具后你会收到预算反馈。",
                    "预算耗尽后必须停止探索，并基于当前证据提交结果。",
                    "先提交 refresh 阶段结果，再调用 deliver(answer=\"stage done\")。",
                ]
            )
        if stage == "navigation_tree":
            return "\n".join(
                [
                    "当前阶段：`navigation_tree`。",
                    "请基于已注入的 `files_tree`，仅在必要时做少量高价值探索，直接产出完整导航树。",
                    "目标是给出稳定的 areas / modules 边界，并让全部候选文件最终可被收敛挂载。",
                    "优先按目录建立 `directory` 模块；只有平铺结构中的少量职责中心文件才用 `file_centered`。",
                    "需要注意观察明显扁平的目录结构：这通常暗示同级文件可能以非目录形式各自承担模块化职责，必要时可批量快速读取这些文件的局部内容来判断。",
                    "若某个 area 内还存在无法稳定细分的剩余文件，请保留一个 `residual` 模块承接，不要把结构化内容直接塞进 fallback。",
                    "显式规则优先表达边界，不要为了追求全覆盖而逐文件穷举。",
                    "本阶段是轻量建模，不要展开大范围项目调研；优先基于现有树结构直接完成划分。",
                    "先调用 `deliver_stage_result(stage='navigation_tree', payload=...)`，再调用 `deliver(answer='stage done')`。",
                ]
            )
        if stage == "navigation_repair":
            current_tree = self._stage_context.get("current_navigation_tree")
            missing_paths_tree = self._stage_context.get("missing_paths_tree")
            validation_feedback = self._stage_context.get("validation_feedback")
            lines = [
                "当前阶段：`navigation_repair`。",
                "请基于现有 `navigation_tree` 做最小修正，补齐遗漏文件，并消除重复/非法挂载。",
                "优先复用已有 area/module；只在明显需要时新增少量 module。",
                "优先修边界：目录模块过宽/过窄、residual 过大、单文件模块过碎，而不是重做全局分类。",
                "若某个平铺目录下的多个同级文件是否应独立成模块仍不清楚，可批量快速读取这些文件的局部内容辅助判断。",
                "修正时优先调整 `include` / `exclude` 规则，避免长串单文件枚举。",
                "本阶段只做局部修复，不要重新调查整个项目。",
                "修正后请重新提交完整 `navigation_tree`。",
                "",
                "## current_navigation_tree",
                json.dumps(current_tree, ensure_ascii=False, indent=2),
                "",
                "## missing_paths_tree",
                json.dumps(missing_paths_tree, ensure_ascii=False, indent=2),
                "",
                "## validation_feedback",
                json.dumps(validation_feedback, ensure_ascii=False, indent=2),
                "",
                "先调用 `deliver_stage_result(stage='navigation_repair', payload=...)`，再调用 `deliver(answer='stage done')`。",
            ]
            return "\n".join(lines)
        if stage == "navigation_incremental":
            current_tree = self._stage_context.get("current_navigation_tree")
            structural_changes = self._stage_context.get("structural_changes")
            validation_feedback = self._stage_context.get("validation_feedback")
            lines = [
                "当前阶段：`navigation_incremental`。",
                "请基于现有 `navigation_tree` 与 dirty file 结构变化，判断本次导航更新级别。",
                "可选决策只有：`no_change`、`module_update`、`area_rebuild`、`full_rebuild`。",
                "如果当前 tree 规则已经足以覆盖本次新增/删除/迁移，直接输出 `no_change`。",
                "如果只涉及少量模块挂载变更，输出 `module_update` 并给出基于当前 tree 最小修正后的完整 `navigation_tree`。",
                "如果变化已经上升到 area 级别，输出 `area_rebuild` 并给出完整 `navigation_tree`。",
                "如果现有 tree 已整体不可信，输出 `full_rebuild`；此时可以不提交新 tree。",
                "本阶段目标是修正挂载与导航，不要主动重写模块认知正文。",
                "",
                "## current_navigation_tree",
                json.dumps(current_tree, ensure_ascii=False, indent=2),
                "",
                "## structural_changes",
                json.dumps(structural_changes, ensure_ascii=False, indent=2),
                "",
                "## validation_feedback",
                json.dumps(validation_feedback, ensure_ascii=False, indent=2),
                "",
                "先调用 `deliver_stage_result(stage='navigation_incremental', payload=...)`，再调用 `deliver(answer='stage done')`。",
            ]
            return "\n".join(lines)
        if stage == "navigation_incremental_repair":
            current_tree = self._stage_context.get("current_navigation_tree")
            structural_changes = self._stage_context.get("structural_changes")
            validation_feedback = self._stage_context.get("validation_feedback")
            lines = [
                "当前阶段：`navigation_incremental_repair`。",
                "上一轮增量导航判定未通过校验，请基于现有 tree 做更保守、更小范围的修正。",
                "优先保持当前 area/module 结构稳定，除非 facts 明确表明必须新增、删除或重建。",
                "若现有 tree 已足以覆盖变化，请直接改判为 `no_change`。",
                "若无法在局部修正下保持结构可信，应提升为 `area_rebuild` 或 `full_rebuild`。",
                "",
                "## current_navigation_tree",
                json.dumps(current_tree, ensure_ascii=False, indent=2),
                "",
                "## structural_changes",
                json.dumps(structural_changes, ensure_ascii=False, indent=2),
                "",
                "## validation_feedback",
                json.dumps(validation_feedback, ensure_ascii=False, indent=2),
                "",
                "先调用 `deliver_stage_result(stage='navigation_incremental_repair', payload=...)`，再调用 `deliver(answer='stage done')`。",
            ]
            return "\n".join(lines)
        if stage == "write_area_modules":
            area_slug = str(self._stage_context.get("area_slug") or "").strip()
            area_name = str(self._stage_context.get("area_display_name") or "").strip()
            area_summary = str(self._stage_context.get("area_summary") or "").strip()
            project_summary = str(self._stage_context.get("project_summary") or "").strip()
            raw_modules = self._stage_context.get("modules")
            modules = raw_modules if isinstance(raw_modules, list) else []
            if not area_slug or not area_name or not modules:
                raise ValueError("write_area_modules 阶段缺少 area 上下文")
            modules_text = json.dumps(modules, ensure_ascii=False, indent=2)
            lines = [
                "当前阶段：`write_area_modules`。",
                f"当前 area: `{area_slug}` / {area_name}",
            ]
            if project_summary:
                lines.append(f"项目背景：{project_summary}")
            if area_summary:
                lines.append(f"area 摘要：{area_summary}")
            lines.extend([
                "",
                "当前需要完成的 modules：",
                modules_text,
                "",
                "任务要求：",
                "- 只处理当前 area 下列出的 modules，不要新增、删除或改名 module slug。",
                "- 对每个已确认的模块，调用 `write_module_annotation(slug, markdown)` 直接写入模块文档。",
                "- 文档中的固定章节必须完整保留；若某章节证据不足，可保留空正文。",
                "- 代码事实优先；文档只能参考不可轻信。",
                "- 当前 area 全部完成后，调用 `mark_area_done(area_slug=..., note=...)`，再调用 `deliver(answer='stage done')`。",
            ])
            return "\n".join(lines)
        raise ValueError(f"不支持的阶段: {stage}")

    def build_tools_schema(
        self, serena_client: SerenaClient, state: LoopState
    ) -> list[dict[str, Any]]:
        filtered = []
        allowed_names = _ALLOWED_SERENA_TOOL_NAMES_BY_STAGE.get(self._active_stage or "", frozenset())
        allow_exploration = not (
            self._active_stage in {"pceignore", "pceignore_refresh", "navigation_tree", "navigation_repair"}
            and self._stage_tool_budget_used >= self._stage_tool_budget_total > 0
        )
        for schema in serena_client.tools_schema:
            try:
                name = schema["function"]["name"]
            except Exception:
                continue
            if allow_exploration and name in allowed_names:
                filtered.append(schema)
        virtual_tools = self._build_virtual_tools()
        self._virtual_tool_names_set = {
            schema["function"]["name"] for schema in virtual_tools
        }
        return filtered + virtual_tools + [self._deliver_tool]

    @property
    def virtual_tool_names(self) -> set[str]:
        return self._virtual_tool_names_set

    def build_no_tool_correction(
        self, state: LoopState, finish_reason: str
    ) -> dict[str, Any]:
        current_stage = self._active_stage or "unknown"
        if current_stage in {"navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair", "pceignore", "pceignore_refresh"}:
            return {
                "role": "user",
                "content": (
                    f"你刚才没有调用工具。当前阶段是 `{current_stage}`。"
                    "如果现有 facts 已足够，请直接调用 `deliver_stage_result(...)` 提交结果，再调用 deliver。"
                    "只有在证据明显不足时才调用少量工具。"
                ),
            }
        return {
            "role": "user",
            "content": (
                f"你刚才没有调用工具。当前阶段是 `{current_stage}`。"
                "请先调用必要工具；阶段完成时必须先提交阶段完成标记，再调用 deliver。"
            ),
        }

    async def on_before_round(self, state: LoopState) -> None:
        if (
            self._active_stage in {"pceignore", "pceignore_refresh", "navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"}
            and self._stage_tool_budget_total > 0
            and self._stage_tool_budget_used >= self._stage_tool_budget_total
            and not self._stage_budget_exhausted_notified
        ):
            self._append(state, {
                "role": "user",
                "content": (
                    "工具预算已耗尽，禁止继续探索。"
                    "请基于当前已掌握的结构与证据直接收口输出。"
                    "不要继续发散调查。"
                ),
            })
            self._stage_budget_exhausted_notified = True

    def _stage_budget_note(self) -> str:
        if self._active_stage not in {"pceignore", "pceignore_refresh", "navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"} or self._stage_tool_budget_total <= 0:
            return ""
        remaining = max(0, self._stage_tool_budget_total - self._stage_tool_budget_used)
        return (
            f"\n\n[预算提示] 你已使用 {self._stage_tool_budget_used}/{self._stage_tool_budget_total} 次工具预算，"
            f"还剩 {remaining} 次。请优先确认最高价值路径。"
        )

    def _render_tree_snapshot(self, root: str, *, max_depth: int, max_lines: int) -> str:
        base = self._project_root / Path(root)
        try:
            rel_base = base.resolve().relative_to(self._project_root)
        except Exception:
            return f"tree_snapshot 失败：路径越界或不存在：{root}"
        if not base.exists() or not base.is_dir():
            return f"tree_snapshot 失败：目录不存在：{root}"

        from .file_discovery import is_ignored, is_probably_text_file

        lines: list[str] = []
        omitted = 0
        root_parts = len(rel_base.parts) if str(rel_base) != "." else 0
        for current_root, dirnames, filenames in os.walk(base, topdown=True):
            current_path = Path(current_root)
            rel = current_path.relative_to(self._project_root).as_posix()
            depth = len(current_path.relative_to(base).parts)
            keep_dirs: list[str] = []
            for dirname in dirnames:
                rel_dir = (current_path / dirname).relative_to(self._project_root).as_posix()
                if is_ignored(self._project_root, rel_dir):
                    continue
                keep_dirs.append(dirname)
            dirnames[:] = keep_dirs

            if depth > max_depth:
                dirnames[:] = []
                continue

            indent = "  " * max(0, len(Path(rel).parts) - root_parts - 1)
            if str(rel_base) == ".":
                if rel == ".":
                    lines.append("- ./")
                else:
                    lines.append(f"{indent}- {Path(rel).name}/")
            else:
                if rel == rel_base.as_posix():
                    lines.append(f"- {Path(rel).name}/")
                else:
                    lines.append(f"{indent}- {Path(rel).name}/")
            if len(lines) >= max_lines:
                omitted += 1
                break

            if depth == max_depth:
                continue

            for filename in sorted(filenames):
                rel_file = (current_path / filename).relative_to(self._project_root).as_posix()
                if is_ignored(self._project_root, rel_file):
                    continue
                kind = "text"
                try:
                    kind = "text" if is_probably_text_file(current_path / filename) else "binary"
                except Exception:
                    kind = "unknown"
                tag = f" [{kind}]" if kind != "text" else ""
                lines.append(f"{indent}  - {filename}{tag}")
                if len(lines) >= max_lines:
                    omitted += 1
                    break
            if len(lines) >= max_lines:
                break

        if omitted > 0:
            lines.append(f"[truncated: omitted {omitted} lines]")
        return "\n".join(lines)

    def _render_batch_file_slice(
        self,
        paths: list[str],
        *,
        start_line: int,
        max_lines: int,
    ) -> str:
        normalized_paths = []
        seen: set[str] = set()
        for raw in paths[:_BATCH_READ_MAX_FILES]:
            path = str(raw or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            normalized_paths.append(path)

        outputs: list[str] = []
        safe_start = max(1, start_line)
        safe_max_lines = max(1, min(_BATCH_READ_MAX_LINES, max_lines))

        for rel_path in normalized_paths:
            abs_path = (self._project_root / rel_path).resolve()
            try:
                abs_path.relative_to(self._project_root)
            except Exception:
                outputs.append(f"## {rel_path}\n[error] 路径越界")
                continue
            if not abs_path.exists() or not abs_path.is_file():
                outputs.append(f"## {rel_path}\n[error] 文件不存在")
                continue
            try:
                lines = abs_path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                outputs.append(f"## {rel_path}\n[error] 读取失败: {exc}")
                continue

            start_idx = min(len(lines), safe_start - 1)
            end_idx = min(len(lines), start_idx + safe_max_lines)
            snippet = lines[start_idx:end_idx]
            outputs.append(
                "\n".join(
                    [f"## {rel_path}", f"[lines {start_idx + 1}-{end_idx}]"]
                    + [f"{idx + 1}: {line}" for idx, line in enumerate(snippet, start=start_idx)]
                )
            )

        return "\n\n".join(outputs) if outputs else "[empty]"

    async def on_deliver(
        self, args: dict[str, Any] | None, state: LoopState
    ) -> DeliverDecision:
        if self._stage_payload is None:
            return DeliverDecision.continue_with(
                {
                    "role": "user",
                    "content": (
                        "你尚未提交阶段完成标记。"
                        "请先调用相应阶段的完成工具，再调用 deliver(answer='stage done')。"
                    ),
                }
            )
        return DeliverDecision.finish(_safe_json_dumps(self._stage_payload))

    async def handle_virtual_tool(
        self, tool_call: Any, state: LoopState
    ) -> dict[str, Any] | None:
        tc_id = _get_tool_call_id(tool_call)
        name = _get_tool_name(tool_call) or "unknown"
        args = _extract_tool_call_args(tool_call) or {}

        def _ok(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        def _err(content: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": content}

        if name == "read_discovery_facts":
            return _ok(_safe_json_dumps(self._discovery_facts))

        if name == "tree_snapshot":
            if self._active_stage not in {"pceignore", "pceignore_refresh"}:
                return _err("tree_snapshot 只能在 pceignore / pceignore_refresh 阶段使用")
            root = str(args.get("root") or ".").strip() or "."
            max_depth = max(1, min(4, int(args.get("max_depth") or 2)))
            max_lines = max(20, min(240, int(args.get("max_lines") or 120)))
            self._stage_tool_budget_used += 1
            return _ok(self._render_tree_snapshot(root, max_depth=max_depth, max_lines=max_lines) + self._stage_budget_note())

        if name == "batch_read_file_slice":
            if self._active_stage not in {"navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"}:
                return _err("batch_read_file_slice 只能在 navigation_tree / navigation_repair / navigation_incremental / navigation_incremental_repair 阶段使用")
            raw_paths = args.get("paths")
            if not isinstance(raw_paths, list):
                return _err("paths 必须是字符串数组")
            start_line = int(args.get("start_line") or 1)
            max_lines = int(args.get("max_lines") or 30)
            self._stage_tool_budget_used += 1
            return _ok(
                self._render_batch_file_slice(
                    [str(item) for item in raw_paths],
                    start_line=start_line,
                    max_lines=max_lines,
                ) + self._stage_budget_note()
            )

        if name == "deliver_stage_result":
            stage = str(args.get("stage") or "").strip()
            payload = args.get("payload")
            if stage != self._active_stage:
                return _err(
                    f"当前阶段是 {self._active_stage!r}，deliver_stage_result.stage 必须与之相同"
                )
            if not isinstance(payload, dict):
                return _err("payload 必须是 object")
            self._stage_payload = payload
            return _ok(f"阶段结果已接收: {stage}")

        if name == "write_module_annotation":
            return await self._handle_write_module_annotation(args, _ok, _err)

        if name == "mark_area_done":
            return self._handle_mark_area_done(args, _ok, _err)

        return _err(f"未知虚拟工具: {name}")

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_discovery_facts",
                    "description": "读取完整 discovery_facts；仅在直接注入 facts 被截断时使用。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "deliver_stage_result",
                    "description": "提交当前阶段的结构化结果。pceignore/navigation_tree 阶段使用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stage": {"type": "string"},
                            "payload": {"type": "object"},
                        },
                        "required": ["stage", "payload"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self._active_stage in {"pceignore", "pceignore_refresh"}:
            tools.append({
                "type": "function",
                "function": {
                    "name": "tree_snapshot",
                    "description": "按路径返回受控、可截断的目录树快照，用于有限目录探索。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "root": {"type": "string"},
                            "max_depth": {"type": "integer"},
                            "max_lines": {"type": "integer"},
                        },
                        "required": ["root"],
                        "additionalProperties": False,
                    },
                },
            })
        if self._active_stage in {"navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"}:
            tools.append({
                "type": "function",
                "function": {
                    "name": "batch_read_file_slice",
                    "description": "批量读取多个文件从指定起始行开始的连续片段；用于扁平目录下快速比较同级文件职责。单次最多 10 个文件、每个文件最多 50 行。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": _BATCH_READ_MAX_FILES,
                            },
                            "start_line": {"type": "integer"},
                            "max_lines": {"type": "integer"},
                        },
                        "required": ["paths"],
                        "additionalProperties": False,
                    },
                },
            })
        if self._active_stage == "write_area_modules":
            tools.extend([
                {
                    "type": "function",
                    "function": {
                        "name": "write_module_annotation",
                        "description": "写入单个模块文档；工具会校验固定章节与覆盖文件路径，失败时返回错误信息。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "slug": {"type": "string"},
                                "markdown": {"type": "string"},
                            },
                            "required": ["slug", "markdown"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mark_area_done",
                        "description": "声明当前 area 已处理完成；Python 侧仍会按真实文件结果校验。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "area_slug": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["area_slug", "note"],
                            "additionalProperties": False,
                        },
                    },
                },
            ])
        return tools

    async def _handle_write_module_annotation(
        self,
        args: dict[str, Any],
        ok_builder: Any,
        err_builder: Any,
    ) -> dict[str, Any]:
        if self._active_stage != "write_area_modules":
            return err_builder("write_module_annotation 只能在 write_area_modules 阶段使用")

        area_slug = str(self._stage_context.get("area_slug") or "").strip()
        modules = self._stage_context.get("modules")
        module_specs = {
            str(item.get("slug") or "").strip(): item
            for item in (modules if isinstance(modules, list) else [])
            if isinstance(item, dict) and str(item.get("slug") or "").strip()
        }
        slug = str(args.get("slug") or "").strip()
        markdown = str(args.get("markdown") or "").strip()
        if slug not in module_specs:
            return err_builder(f"模块 `{slug}` 不属于当前 area `{area_slug}`")
        if not markdown:
            return err_builder("markdown 不能为空")

        spec = module_specs[slug]
        expected_paths = [
            str(path).strip()
            for path in spec.get("file_paths", [])
            if str(path).strip()
        ]
        errors = validate_module_annotation_markdown(
            markdown,
            expected_file_paths=expected_paths,
            require_core_responsibility=True,
        )
        if errors:
            return err_builder("写入校验失败: " + " | ".join(errors))

        modules_dir = self._project_root / ".pce" / "annotations" / "modules"
        await self._write_text_atomic(
            modules_dir / f"{slug}.md",
            markdown.rstrip() + "\n",
        )
        self._written_module_slugs_by_area.setdefault(area_slug, set()).add(slug)
        return ok_builder(f"模块文档已写入: {slug}")

    def _handle_mark_area_done(
        self,
        args: dict[str, Any],
        ok_builder: Any,
        err_builder: Any,
    ) -> dict[str, Any]:
        if self._active_stage != "write_area_modules":
            return err_builder("mark_area_done 只能在 write_area_modules 阶段使用")
        current_area_slug = str(self._stage_context.get("area_slug") or "").strip()
        area_slug = str(args.get("area_slug") or "").strip()
        note = str(args.get("note") or "").strip()
        if area_slug != current_area_slug:
            return err_builder(
                f"当前 area 是 `{current_area_slug}`，mark_area_done.area_slug 必须与之相同"
            )
        if not note:
            return err_builder("mark_area_done.note 不能为空")
        written = sorted(self._written_module_slugs_by_area.get(area_slug, set()))
        self._stage_payload = {
            "area_slug": area_slug,
            "written_modules": written,
            "note": note,
        }
        return ok_builder(f"当前 area 已标记完成: {area_slug}")

    async def _invoke_serena(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
        state: LoopState | None = None,
    ) -> dict[str, Any]:
        tool_name = _get_tool_name(tool_call) or "unknown"
        result = await super()._invoke_serena(tool_call, serena_client, state=state)
        if (
            self._active_stage in {"pceignore", "pceignore_refresh", "navigation_tree", "navigation_repair", "navigation_incremental", "navigation_incremental_repair"}
            and tool_name in _ALLOWED_SERENA_TOOL_NAMES_BY_STAGE[self._active_stage]
            and self._stage_tool_budget_total > 0
        ):
            self._stage_tool_budget_used += 1
            result["content"] = str(result.get("content") or "") + self._stage_budget_note()
        return result

    @staticmethod
    async def _write_text_atomic(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        await asyncio.to_thread(tmp_path.write_text, content, "utf-8")
        try:
            await asyncio.to_thread(os.replace, tmp_path, path)
        finally:
            if tmp_path.exists():
                await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
