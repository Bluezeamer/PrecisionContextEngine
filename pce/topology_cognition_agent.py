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
from .init_cognition_limits import PCEIGNORE_STAGE_TOOL_BUDGET
from .module_annotation_contract import validate_module_annotation_markdown
from .prompt_guard import fit_text_to_budget
from .serena_client import SerenaClient

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SECONDS = 180.0
_ALLOWED_SERENA_TOOL_NAMES_BY_STAGE: dict[str, frozenset[str]] = {
    "pceignore": frozenset({"read_file", "search_for_pattern"}),
    "pceignore_refresh": frozenset({"read_file", "search_for_pattern"}),
    "navigation_tree": frozenset({"search_for_pattern", "find_file", "get_symbols_overview", "find_symbol", "find_referencing_symbols", "read_file"}),
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
        if stage != "write_area_modules":
            self._stage_context = {}
        if stage in {"pceignore", "pceignore_refresh"}:
            self._stage_tool_budget_total = PCEIGNORE_STAGE_TOOL_BUDGET
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
        discovery_text = json.dumps(
            self._discovery_facts, ensure_ascii=False, indent=2
        )
        discovery_text = fit_text_to_budget(
            self._model,
            discovery_text,
            token_budget=14000,
            notice="\n... [discovery facts 已截断，可用 read_discovery_facts 取回更多内容] ...\n",
            min_chars=1800,
        )
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
                "- `areas[*].modules[*]` 字段：`slug`, `display_name`, `summary`, `file_paths`。",
                "- `areas[*].modules[*].summary` 必须是一句 module 入口介绍，用于快速导航。",
                "- 恰好 1 个 fallback area；每个文件只能归属一个 module；每个 module 只能归属一个 area。",
                "- 不要单独输出 module_catalog，也不要把目录名机械等同于 area/module。",
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
                    "请基于已注入的目录结构、统计与 .gitignore 事实，必要时再做极少量探索，输出保守可用的 `.pce/pceignore` 规则。",
                    "本阶段最多允许有限工具探索；每次调用工具后你会收到预算反馈。",
                    "预算耗尽后必须停止探索，并基于当前证据提交结果。",
                    "先提交 pceignore 阶段结果，再调用 deliver(answer=\"stage done\")。",
                ]
            )
        if stage == "pceignore_refresh":
            return "\n".join(
                [
                    "当前阶段：`pceignore_refresh`。",
                    "请基于已注入的 dirty files、当前 `.pce/pceignore`、目录摘要与统计事实，判断当前 ignore 边界是否需要增量更新。",
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
                    "请基于已注入 discovery facts，必要时再探索代码，直接产出完整导航树。",
                    "你需要同时完成项目概括、areas 划分、modules 划分，以及 area/module 的导航摘要。",
                    "先调用 `deliver_stage_result(stage='navigation_tree', payload=...)`，再调用 `deliver(answer='stage done')`。",
                ]
            )
        if stage == "write_area_modules":
            area_slug = str(self._stage_context.get("area_slug") or "").strip()
            area_name = str(self._stage_context.get("area_display_name") or "").strip()
            area_summary = str(self._stage_context.get("area_summary") or "").strip()
            project_summary = str(self._stage_context.get("project_summary") or "").strip()
            raw_modules = self._stage_context.get("modules")
            modules = raw_modules if isinstance(raw_modules, list) else []
            if not area_slug or not area_name or not modules:
                raise ValueError("write_area_modules 阶段缺少 area 上下文")
            modules_text = fit_text_to_budget(
                self._model,
                json.dumps(modules, ensure_ascii=False, indent=2),
                token_budget=5000,
                notice="\n... [当前 area 的 modules 清单已截断，请优先围绕已给 facts 与代码证据推进] ...\n",
                min_chars=1200,
            )
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
            self._active_stage in {"pceignore", "pceignore_refresh"}
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
        return {
            "role": "user",
            "content": (
                f"你刚才没有调用工具。当前阶段是 `{current_stage}`。"
                "请先调用必要工具；阶段完成时必须先提交阶段完成标记，再调用 deliver。"
            ),
        }

    async def on_before_round(self, state: LoopState) -> None:
        if (
            self._active_stage in {"pceignore", "pceignore_refresh"}
            and self._stage_tool_budget_total > 0
            and self._stage_tool_budget_used >= self._stage_tool_budget_total
            and not self._stage_budget_exhausted_notified
        ):
            self._append(state, {
                "role": "user",
                "content": (
                    "工具预算已耗尽，禁止继续探索。"
                    "请基于当前已掌握的目录结构与证据，输出最保守可用的结果。"
                    "不要误伤可能属于源码、配置、脚本、测试、关键文档的路径。"
                ),
            })
            self._stage_budget_exhausted_notified = True

    def _stage_budget_note(self) -> str:
        if self._active_stage not in {"pceignore", "pceignore_refresh"} or self._stage_tool_budget_total <= 0:
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
            self._active_stage in {"pceignore", "pceignore_refresh"}
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
