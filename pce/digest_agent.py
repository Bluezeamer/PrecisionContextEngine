"""DigestAgent — 认知整合 Agent。

在 pce_init / pce_sync 完成 Serena 索引重建后，将 InsightCache 积累的细粒度观察
和模块级代码变化事实，内化到 `.pce/annotations/modules/*.md` 中。

设计原则：
- 一次完整 ReAct 循环，Agent 自主决定探索和更新顺序
- 任务清单（DigestTaskList）注入 system prompt，作为无遗漏保障，不约束执行顺序
- 虚拟工具（任务追踪 + annotation 读写）由 Python 拦截，不透传给 Serena
- 失败为 best-effort：异常只记 warning，不影响 init/sync 主流程
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .base_agent import (
    BaseReActAgent,
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
from .memory import delete_file_baseline, load_index, save_file_baseline
from .models import ModuleDigestDelta, SymbolFact
from .prompt_guard import build_prompt_budget, estimate_input_tokens
from .serena_client import SerenaClient
from .staging import DirtyState

logger = logging.getLogger(__name__)

# annotation 固定章节，不可删除
FIXED_SECTION_HEADINGS: tuple[str, ...] = (
    "覆盖文件",
    "核心职责",
    "关键流程",
    "外部协作",
    "风险与约束",
)
LEGACY_OPTIONAL_SECTION_HEADINGS: tuple[str, ...] = (
    "关键符号",
)
ALLOWED_EXTENSION_SECTION_PREFIXES: tuple[str, ...] = (
    "专题：",
    "变更专题：",
    "设计约束：",
    "已知问题：",
)

_DIGEST_TASKS_REL = Path(".pce") / "digest_tasks.json"
_DEFAULT_MAX_SECONDS = 300.0
_DEFAULT_DIGEST_BATCH_SIZE = 4

_REACT_FAILURE_MESSAGES: dict[str, str] = {
    REACT_TIMEOUT_BUDGET: "DigestAgent ReAct 循环超时，已达最大时间预算",
    REACT_NO_TOOL_EXHAUSTED: "DigestAgent 连续未产生 tool_calls，已终止",
    REACT_LENGTH_EXHAUSTED: "DigestAgent 输出连续被截断，已放弃本轮",
    REACT_TIMEOUT: "DigestAgent 模型调用超时，重试次数耗尽",
    REACT_LLM_EXHAUSTED: "DigestAgent 模型降级链耗尽，已终止",
}


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _get_digest_batch_size() -> int:
    raw = os.getenv("PCE_DIGEST_BATCH_SIZE", str(_DEFAULT_DIGEST_BATCH_SIZE)).strip()
    try:
        value = int(raw)
    except ValueError:
        logger.warning("PCE_DIGEST_BATCH_SIZE 非法，回退默认值: %s", raw)
        return _DEFAULT_DIGEST_BATCH_SIZE
    return max(1, value)


def _chunk_digest_items(
    items: list[DigestTaskItem],
    *,
    batch_size: int,
) -> list[list[DigestTaskItem]]:
    if batch_size <= 0:
        batch_size = _DEFAULT_DIGEST_BATCH_SIZE
    return [items[idx: idx + batch_size] for idx in range(0, len(items), batch_size)]


def _module_slug_from_name(name: str) -> str:
    normalized = " ".join(name.strip().split()) or "unnamed-module"
    return normalized.lower().replace(" ", "-")


async def _read_file(path: Path) -> str:
    return await asyncio.to_thread(path.read_text, encoding="utf-8")


async def _write_file_atomic(path: Path, content: str) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    await asyncio.to_thread(_write)



# ---------------------------------------------------------------------------
# 任务数据结构
# ---------------------------------------------------------------------------


@dataclass
class DigestTaskItem:
    id: str
    kind: str  # "module"
    status: str  # "pending" | "done" | "skipped"
    note: str | None = None
    module_slug: str | None = None
    digest_delta: ModuleDigestDelta | None = None


@dataclass
class DigestTaskList:
    items: list[DigestTaskItem]
    warnings: list[str]
    created_at: datetime

    @staticmethod
    def _summarize_item(item: DigestTaskItem) -> dict[str, Any]:
        delta = item.digest_delta
        changed_paths = [str(file_fact.path) for file_fact in delta.changed_files] if delta else []
        insight_scopes = list(dict.fromkeys(insight.scope for insight in delta.related_insights)) if delta else []
        changed_preview = changed_paths[:5]
        insight_preview = insight_scopes[:5]
        if len(changed_paths) > 5:
            changed_preview.append(f"... (+{len(changed_paths) - 5} more)")
        if len(insight_scopes) > 5:
            insight_preview.append(f"... (+{len(insight_scopes) - 5} more)")

        return {
            "id": item.id,
            "kind": item.kind,
            "status": item.status,
            "note": item.note,
            "module_slug": item.module_slug,
            "module_name": delta.module_name if delta is not None else None,
            "change_scope_hint": delta.change_scope_hint if delta is not None else "agent_decide",
            "changed_files_count": len(delta.changed_files) if delta is not None else 0,
            "related_insights_count": len(delta.related_insights) if delta is not None else 0,
            "external_context_count": len(delta.external_context) if delta is not None else 0,
            "insight_only": bool(delta and not delta.changed_files and delta.related_insights),
            "changed_paths_preview": changed_preview,
            "insight_scopes_preview": insight_preview,
        }

    def all_resolved(self) -> bool:
        return not any(item.status == "pending" for item in self.items)

    def pending_items(self) -> list[DigestTaskItem]:
        return [item for item in self.items if item.status == "pending"]

    def to_dict(self, *, include_digest_delta: bool = True) -> dict[str, Any]:
        serialized_items: list[dict[str, Any]] = []
        for item in self.items:
            payload = asdict(item)
            if include_digest_delta and item.digest_delta is not None:
                payload["digest_delta"] = item.digest_delta.model_dump(mode="json")
            elif not include_digest_delta:
                payload.pop("digest_delta", None)
            serialized_items.append(payload)
        return {
            "created_at": self.created_at.isoformat(),
            "warnings": list(self.warnings),
            "items": serialized_items,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "warnings": list(self.warnings),
            "items": [self._summarize_item(item) for item in self.items],
        }

    async def save(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(include_digest_delta=True), ensure_ascii=False, indent=2) + "\n"
        await _write_file_atomic(path, payload)

    async def delete(self, path: Path) -> None:
        await asyncio.to_thread(path.unlink, True)


# ---------------------------------------------------------------------------
# DigestPlanner：构建任务清单
# ---------------------------------------------------------------------------


class DigestPlanner:
    """基于 DigestDelta 构建模块级 DigestTaskList。"""

    def __init__(self, project_root: Path, insight_cache: InsightCache) -> None:
        self._project_root = project_root.resolve()
        self._insight_cache = insight_cache

    async def build(self, dirty_state: DirtyState) -> DigestTaskList:
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
        warnings: list[str] = []
        for path in sorted(dirty_paths - covered_paths):
            warnings.append(f"dirty 文件暂未命中 module registry，后续需补归属: {path}")

        items = [
            DigestTaskItem(
                id=f"module:{delta.module_slug}",
                kind="module",
                status="pending",
                module_slug=delta.module_slug,
                digest_delta=delta,
            )
            for delta in module_deltas
        ]

        return DigestTaskList(items=items, warnings=warnings, created_at=_utc_now())


# ---------------------------------------------------------------------------
# DigestAgent：ReAct 循环执行认知整合
# ---------------------------------------------------------------------------


class DigestAgent(BaseReActAgent):
    """认知整合专用 ReAct Agent。

    继承 BaseReActAgent，通过钩子注入 digest 特有行为：
    - 虚拟工具（任务追踪 + annotation 读写）由 Python 拦截
    - Serena 只读工具透传，供 Agent 按需探索代码库
    """

    _completion_temperature = 0.1
    _enable_budget_warning = True

    def __init__(
        self,
        project_root: Path,
        task_list_path: Path,
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
        self._task_list_path = task_list_path
        self._virtual_tools = self._build_virtual_tools()
        self._virtual_tool_names_set = {
            schema["function"]["name"] for schema in self._virtual_tools
        }
        self._task_list: DigestTaskList | None = None

    async def run(
        self,
        *,
        task_list: DigestTaskList,
        serena_client: SerenaClient,
    ) -> dict[str, Any]:
        self._task_list = task_list

        if not task_list.items:
            return {
                "summary": "",
                "resolved_tasks": 0,
                "pending_tasks": 0,
                "warnings": list(task_list.warnings),
            }

        system_prompt = self._build_system_prompt(task_list)
        system_prompt = self._guard_system_prompt(task_list, serena_client, system_prompt)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "请开始认知整合。先用 read_task_list 了解任务，"
                    "再按需通过 Serena 工具探索代码，读取并更新 annotation，"
                    "标记每个任务状态，最后调用 deliver(summary=...) 结束。"
                ),
            },
        ]

        summary, _ = await self.run_loop(messages, serena_client)
        # 检查是否为异常终止
        failure = _REACT_FAILURE_MESSAGES.get(summary)
        if failure is not None:
            raise RuntimeError(failure)

        resolved = sum(1 for item in task_list.items if item.status != "pending")
        return {
            "summary": summary,
            "resolved_tasks": resolved,
            "pending_tasks": len(task_list.pending_items()),
            "warnings": list(task_list.warnings),
        }

    def _build_task_summary_payload(
        self,
        task_list: DigestTaskList,
        *,
        detail: str,
    ) -> dict[str, Any]:
        summary = task_list.to_summary_dict()
        if detail == "full":
            return summary
        if detail == "compact":
            return {
                "created_at": summary["created_at"],
                "warnings": summary["warnings"][:5],
                "items": [
                    {
                        "id": item["id"],
                        "module_slug": item["module_slug"],
                        "status": item["status"],
                        "changed_files_count": item["changed_files_count"],
                        "related_insights_count": item["related_insights_count"],
                        "insight_only": item["insight_only"],
                        "changed_paths_preview": item["changed_paths_preview"][:2],
                    }
                    for item in summary["items"]
                ],
            }
        return {
            "created_at": summary["created_at"],
            "task_count": len(summary["items"]),
            "warnings_count": len(summary["warnings"]),
            "items": [
                {
                    "id": item["id"],
                    "module_slug": item["module_slug"],
                    "changed_files_count": item["changed_files_count"],
                    "related_insights_count": item["related_insights_count"],
                    "insight_only": item["insight_only"],
                }
                for item in summary["items"][:12]
            ],
        }

    def _build_system_prompt(self, task_list: DigestTaskList, *, detail: str = "full") -> str:
        task_json = json.dumps(
            self._build_task_summary_payload(task_list, detail=detail),
            ensure_ascii=False,
            indent=2,
        )
        fixed = "、".join(FIXED_SECTION_HEADINGS)
        extension_prefixes = "、".join(f"`{prefix}...`" for prefix in ALLOWED_EXTENSION_SECTION_PREFIXES)
        return "\n".join(
            [
                "你是 PCE 的 DigestAgent，负责把历史 insight 和代码变更内化到模块 annotation 文档。",
                "",
                "## 目标",
                "1. 先用 read_task_list 查看模块级任务摘要，再用 read_digest_delta(task_id) 读取完整事实包。",
                "2. 按需通过 Serena 工具探索代码获取最新证据。",
                "3. 优先用 read_annotation + write_annotation_and_mark_done 一次性更新 annotation 并完成任务。",
                "4. 只有确实需要分步编辑时，才单独使用 write_annotation_patch；编辑完成后立刻 mark_task_done。",
                "5. 完成或跳过每个任务后，调用 mark_task_done / mark_task_skipped 记录结论。",
                "6. 所有任务处理完毕后，调用 deliver(summary=...) 结束。",
                "",
                "## 固定约束",
                f"- annotation 固定章节不可删除：{fixed}",
                f"- 可新增额外 `##` 章节，但标题必须使用受控前缀：{extension_prefixes}",
                "- `replace` 按 `##` 级标题定位，替换整个章节正文（包含想保留的内容）。",
                "- `rewrite` 提交完整 Markdown，必须显式保留全部 6 个固定章节。",
                "- `append` 在指定章节末尾追加内容；target=null 时追加到文档末尾。",
                "- 每个任务必须有明确结论（done 或 skipped），不允许静默跳过。",
                "- mark_task_done / mark_task_skipped 的 note 字段必须填写。",
                "- `digest_delta` 是模块级事实包，包含 annotation 基线、相关 insight、changed_files、patch_blocks 与必要的外部上下文提示。",
                "",
                "## 工作策略",
                "- 先用 read_task_list 看摘要，再对当前要处理的任务调用 read_digest_delta(task_id)。",
                "- 优先消化 digest_delta 中的 annotation_baseline、related_insights、changed_files、patch_blocks。",
                "- 若 changed_files_count=0 且 related_insights_count>0，默认先基于 annotation_baseline 和 related_insights 直接修正文档，不要先调用 Serena。",
                "- 若 digest_delta 已足够支撑认知修正，尽量直接 deliver 或 write_annotation_and_mark_done，不要无谓探索。",
                "- 若 `change_scope_hint=route`，默认视为导航/挂载变化；尽量少改模块正文，只在覆盖文件或明显失真时做最小修正。",
                "- 若 `change_scope_hint=module`，默认优先更新该模块正文，不要为了求完整而扩散到其它模块。",
                "- 若 `change_scope_hint=agent_decide`，先基于事实判断影响层级，再决定是否需要实质改写模块正文。",
                "- 若任务已给出 module_slug，先 read_annotation；只有 digest_delta 证据不足时再调用 Serena。",
                "- 处理代码文件时优先使用 get_symbols_overview / find_symbol / search_for_pattern，避免大段 read_file。",
                "- 同一 module_slug 的多个任务尽量连续处理，减少重复探索与重复写入。",
                "- 一旦证据足够，立即调用 write_annotation_and_mark_done 或 mark_task_done 收尾，不要把收尾拖到最后。",
                "- 若时间预算所剩不多且证据仍不足，调用 mark_task_skipped 写清原因，不要继续扩散探索。",
                "",
                "## 输入说明",
                "- `changed_paths_preview` 是当前模块改动文件的预览，优先据此定位需要核对的 patch blocks。",
                "- `change_scope_hint` 是轻量预判提示，不是强制命令；若实际证据冲突，以代码事实为准。",
                "- `related_insights_count>0` 表示该模块存在待内化的短期认知，需结合 annotation_baseline 校验是否已过时。",
                "- `insight_only=true` 表示当前没有代码差异文件，重点是把已有 insight 稳定沉淀到 annotation。",
                "- `changed_files_count=0` 不代表该任务无价值，通常表示“仅有 insight 待沉淀”的模块任务。",
                "",
                "## 当前任务清单",
                task_json,
            ]
        )

    def _guard_system_prompt(
        self,
        task_list: DigestTaskList,
        serena_client: SerenaClient,
        initial_prompt: str,
    ) -> str:
        budget = build_prompt_budget()
        digest_deliver = {
            "type": "function",
            "function": {
                "name": "deliver",
                "description": "提交本轮 digest 总结并结束循环。所有任务 resolved 后才能调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        }
        tools_schema = serena_client.tools_schema + self._virtual_tools + [digest_deliver]
        user_content = (
            "请开始认知整合。先用 read_task_list 了解任务，"
            "再按需通过 Serena 工具探索代码，读取并更新 annotation，"
            "标记每个任务状态，最后调用 deliver(summary=...) 结束。"
        )

        prompt = initial_prompt
        for detail in ("full", "compact", "minimal"):
            prompt = self._build_system_prompt(task_list, detail=detail)
            tokens = estimate_input_tokens(
                self._model,
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                tools_schema,
            )
            if tokens <= budget.hard_input_budget:
                if detail != "full":
                    logger.warning("digest system prompt 触发降级: detail=%s tokens=%d", detail, tokens)
                return prompt

        logger.warning("digest system prompt 触发最终降级: 使用极简任务摘要")
        minimal_payload = {
            "task_count": len(task_list.items),
            "pending_count": len(task_list.pending_items()),
            "warnings_count": len(task_list.warnings),
        }
        return "\n".join(
            [
                "你是 PCE 的 DigestAgent，负责把历史 insight 和代码变更内化到模块 annotation 文档。",
                "",
                "## 目标",
                "1. 先用 read_task_list 查看任务摘要，再用 read_digest_delta(task_id) 按需读取完整事实包。",
                "2. 证据足够时，优先用 write_annotation_and_mark_done 收尾。",
                "3. 若证据不足，调用 mark_task_skipped 写明原因。",
                "",
                "## 当前任务概览",
                json.dumps(minimal_payload, ensure_ascii=False, indent=2),
            ]
        )

    # ── BaseReActAgent 钩子实现 ────────────────────────────────────────────

    def build_tools_schema(
        self, serena_client: SerenaClient, state: LoopState
    ) -> list[dict[str, Any]]:
        # DigestAgent 使用自定义 deliver schema（参数为 summary 而非 answer）
        digest_deliver = {
            "type": "function",
            "function": {
                "name": "deliver",
                "description": "提交本轮 digest 总结并结束循环。所有任务 resolved 后才能调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "本次 digest 的整合总结",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }
        return serena_client.tools_schema + self._virtual_tools + [digest_deliver]

    @property
    def virtual_tool_names(self) -> set[str]:
        return self._virtual_tool_names_set

    async def on_before_round(self, state: LoopState) -> None:
        logger.info(
            "DigestAgent round=%d elapsed=%.1fs pending=%d",
            state.round_num,
            state.elapsed,
            len(self._task_list.pending_items()) if self._task_list else -1,
        )

    async def on_budget_warning(self, state: LoopState) -> None:
        self._append(state, {
            "role": "user",
            "content": (
                f"时间预算仅剩约 {int(state.remaining)} 秒。请停止扩散探索：\n"
                "1) 对证据已足够的任务，立即用 write_annotation_and_mark_done 或 mark_task_done 收尾；\n"
                "2) 对暂时证据不足的任务，直接用 mark_task_skipped 说明原因；\n"
                "3) 不要再做大范围 Serena 探索，全部任务 resolved 后再 deliver。"
            ),
        })

    def build_no_tool_correction(
        self, state: LoopState, finish_reason: str
    ) -> dict[str, Any]:
        return {
            "role": "user",
            "content": (
                "你刚才没有调用工具。DigestAgent 必须通过工具处理任务，"
                "请先用 read_task_list 查看任务，"
                "然后优先用 read_annotation / write_annotation_and_mark_done / mark_task_skipped 收尾。"
            ),
        }

    async def on_deliver(
        self, args: dict[str, Any] | None, state: LoopState
    ) -> DeliverDecision:
        assert self._task_list is not None
        summary = str((args or {}).get("summary") or "").strip()
        if not summary:
            return DeliverDecision.continue_with({
                "role": "user",
                "content": "deliver.summary 不能为空，请重新调用 deliver。",
            })

        if not self._task_list.all_resolved():
            if not state.extra.get("deliver_guard_used", False):
                pending = [
                    self._task_list._summarize_item(item)
                    for item in self._task_list.pending_items()
                ]
                state.extra["deliver_guard_used"] = True
                return DeliverDecision.continue_with({
                    "role": "user",
                    "content": (
                        f"还有 {len(pending)} 个未解决的任务，请继续处理后再 deliver。\n"
                        "优先使用 write_annotation_and_mark_done 收尾；"
                        "若证据不足则调用 mark_task_skipped，禁止继续空转：\n"
                        f"{json.dumps(pending, ensure_ascii=False, indent=2)}"
                    ),
                })
            # 追问后仍有遗漏，记录 warning 但允许退出
            for item in self._task_list.pending_items():
                self._task_list.warnings.append(
                    f"任务未被 Agent 处理即结束: {item.id} ({item.kind})"
                )

        return DeliverDecision.finish(summary)

    async def handle_virtual_tool(
        self, tool_call: Any, state: LoopState
    ) -> dict[str, Any] | None:
        assert self._task_list is not None

        tc_id = _get_tool_call_id(tool_call)
        name = _get_tool_name(tool_call) or "unknown"
        args: dict[str, Any] = _extract_tool_call_args(tool_call) or {}

        def _err(msg: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": msg}

        def _ok(msg: str) -> dict[str, Any]:
            return {"tool_call_id": tc_id, "name": name, "content": msg}

        if name == "read_task_list":
            return _ok(_safe_json_dumps(self._task_list.to_summary_dict()))

        if name == "read_digest_delta":
            task_id = str(args.get("task_id") or "").strip()
            if not task_id:
                return _err("task_id 不能为空")
            task = self._find_task(task_id)
            if task is None:
                return _err(f"未找到任务: {task_id}")
            if task.digest_delta is None:
                return _err(f"任务无 digest_delta: {task_id}")
            return _ok(_safe_json_dumps(task.digest_delta.model_dump(mode="json")))

        if name in {"mark_task_done", "mark_task_skipped"}:
            task_id = str(args.get("task_id") or "").strip()
            note = str(args.get("note") or "").strip()
            if not task_id:
                return _err("task_id 不能为空")
            if not note:
                return _err("note 不能为空")
            task = self._find_task(task_id)
            if task is None:
                return _err(f"未找到任务: {task_id}")
            await self._update_task_status(
                task,
                "done" if name == "mark_task_done" else "skipped",
                note,
            )
            return _ok(f"任务 {task_id} 已标记为 {task.status}")

        if name == "read_annotation":
            module_slug = str(args.get("module_slug") or "").strip()
            if not module_slug:
                return _err("module_slug 不能为空")
            content = await self._load_annotation(module_slug)
            return _ok(content)

        if name == "write_annotation_patch":
            module_slug = str(args.get("module_slug") or "").strip()
            operation = str(args.get("operation") or "").strip().lower()
            target = args.get("target")
            target_str = (
                str(target).strip()
                if isinstance(target, str) and target.strip()
                else None
            )
            content = str(args.get("content") or "")
            if not module_slug:
                return _err("module_slug 不能为空")
            try:
                await self._write_annotation_patch(
                    module_slug=module_slug,
                    operation=operation,
                    target=target_str,
                    content=content,
                )
                return _ok(
                    f"annotation 已更新: module={module_slug}, operation={operation}, "
                    f"target={target_str!r}"
                )
            except Exception as e:
                return _err(f"annotation 更新失败: {e}")

        if name == "write_annotation_and_mark_done":
            task_id = str(args.get("task_id") or "").strip()
            note = str(args.get("note") or "").strip()
            module_slug = str(args.get("module_slug") or "").strip()
            operation = str(args.get("operation") or "").strip().lower()
            target = args.get("target")
            target_str = (
                str(target).strip()
                if isinstance(target, str) and target.strip()
                else None
            )
            content = str(args.get("content") or "")
            if not task_id:
                return _err("task_id 不能为空")
            if not note:
                return _err("note 不能为空")
            task = self._find_task(task_id)
            if task is None:
                return _err(f"未找到任务: {task_id}")
            effective_module_slug = module_slug or (task.module_slug or "")
            if not effective_module_slug:
                return _err("module_slug 不能为空，且任务自身未提供 module_slug")
            try:
                await self._write_annotation_patch(
                    module_slug=effective_module_slug,
                    operation=operation,
                    target=target_str,
                    content=content,
                )
                await self._update_task_status(task, "done", note)
                return _ok(
                    f"annotation 已更新并完成任务: task={task_id}, "
                    f"module={effective_module_slug}, operation={operation}, target={target_str!r}"
                )
            except Exception as e:
                return _err(f"更新 annotation 并完成任务失败: {e}")

        return _err(f"未知虚拟工具: {name}")

    # ── 私有方法 ─────────────────────────────────────────────────────────

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_task_list",
                    "description": "读取当前 digest 任务摘要（不含完整 digest_delta 正文）。",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_digest_delta",
                    "description": "读取指定任务的完整 digest_delta 事实包。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务 ID"},
                        },
                        "required": ["task_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_done",
                    "description": "将任务标记为 done，并记录处理结论（note 必填）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务 ID"},
                            "note": {"type": "string", "description": "完成说明（必填）"},
                        },
                        "required": ["task_id", "note"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mark_task_skipped",
                    "description": "将任务标记为 skipped，并记录跳过原因（note 必填）。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务 ID"},
                            "note": {"type": "string", "description": "跳过原因（必填）"},
                        },
                        "required": ["task_id", "note"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_annotation",
                    "description": "读取指定模块的 annotation Markdown 文档。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_slug": {
                                "type": "string",
                                "description": "模块 slug（对应 .pce/annotations/modules/{slug}.md）",
                            },
                        },
                        "required": ["module_slug"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_annotation_patch",
                    "description": (
                        "仅在需要分步编辑、暂时还不打算结束任务时使用。"
                        "常规完成任务请优先改用 write_annotation_and_mark_done。"
                        "\n"
                        "更新模块 annotation 文档。"
                        "append: 追加到章节末尾（target=null 追加到文档末尾）；"
                        "replace: 替换整个章节正文（需在 content 中包含想保留的内容）；"
                        "rewrite: 提交完整文档（必须保留 6 个固定章节）。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_slug": {
                                "type": "string",
                                "description": "模块 slug",
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["append", "replace", "rewrite"],
                                "description": "操作类型",
                            },
                            "target": {
                                "type": ["string", "null"],
                                "description": "目标 `##` 章节标题；rewrite 或末尾追加时为 null",
                            },
                            "content": {
                                "type": "string",
                                "description": "待写入的 Markdown 内容",
                            },
                        },
                        "required": ["module_slug", "operation", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_annotation_and_mark_done",
                    "description": (
                        "推荐默认使用：先更新 annotation，再将任务原子化标记为 done。"
                        "适合“证据已足够，准备收尾”的场景，可避免只写文档却忘记 mark done。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string", "description": "任务 ID"},
                            "note": {"type": "string", "description": "完成说明（必填）"},
                            "module_slug": {
                                "type": "string",
                                "description": "模块 slug；省略时默认使用任务自身的 module_slug",
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["append", "replace", "rewrite"],
                                "description": "annotation 更新类型",
                            },
                            "target": {
                                "type": ["string", "null"],
                                "description": "目标 `##` 章节标题；rewrite 或末尾追加时为 null",
                            },
                            "content": {
                                "type": "string",
                                "description": "待写入的 Markdown 内容",
                            },
                        },
                        "required": ["task_id", "note", "operation", "content"],
                    },
                },
            },
        ]

    def _find_task(self, task_id: str) -> DigestTaskItem | None:
        assert self._task_list is not None
        return next((item for item in self._task_list.items if item.id == task_id), None)

    async def _update_task_status(
        self,
        task: DigestTaskItem,
        status: str,
        note: str,
    ) -> None:
        assert self._task_list is not None
        prev_status = task.status
        prev_note = task.note
        task.status = status
        task.note = note
        try:
            await self._task_list.save(self._task_list_path)
        except Exception:
            task.status = prev_status
            task.note = prev_note
            raise

    async def _write_annotation_patch(
        self,
        *,
        module_slug: str,
        operation: str,
        target: str | None,
        content: str,
    ) -> None:
        original = await self._load_annotation(module_slug)
        updated = self._apply_patch(original, operation, target, content, module_slug)
        await _write_file_atomic(self._annotation_path(module_slug), updated)

    # ------------------------------------------------------------------
    # Annotation 解析 / patch / 渲染
    # ------------------------------------------------------------------

    def _apply_patch(
        self,
        original: str,
        operation: str,
        target: str | None,
        content: str,
        module_slug: str,
    ) -> str:
        if operation not in {"append", "replace", "rewrite"}:
            raise ValueError(f"不支持的 operation: {operation!r}")

        if operation == "rewrite":
            # 校验固定章节存在；允许保留旧版的「关键符号」章节。
            parsed = self._parse_annotation(content, module_slug, ensure_fixed=False)
            headings = {s["heading"] for s in parsed["sections"]}
            missing = [h for h in FIXED_SECTION_HEADINGS if h not in headings]
            if missing:
                raise ValueError(f"rewrite 缺少固定章节: {missing}")
            invalid = [
                h
                for h in headings
                if h not in FIXED_SECTION_HEADINGS
                and h not in LEGACY_OPTIONAL_SECTION_HEADINGS
                and not self._is_allowed_extra_heading(h)
            ]
            if invalid:
                raise ValueError(f"rewrite 包含不受支持的扩展章节标题: {invalid}")
            return self._render_annotation(parsed)

        parsed = self._parse_annotation(original, module_slug, ensure_fixed=True)
        sections = parsed["sections"]

        if operation == "append" and target is None:
            # 追加到文档末尾
            rendered = self._render_annotation(parsed).rstrip()
            suffix = content.strip()
            return (rendered + "\n\n" + suffix + "\n") if suffix else (rendered + "\n")

        if target is None:
            raise ValueError(f"{operation} 操作必须提供 target（章节标题）")

        if (
            target not in FIXED_SECTION_HEADINGS
            and target not in LEGACY_OPTIONAL_SECTION_HEADINGS
            and not self._is_allowed_extra_heading(target)
        ):
            raise ValueError(
                "扩展章节标题必须使用受控前缀: "
                + ", ".join(ALLOWED_EXTENSION_SECTION_PREFIXES)
            )

        idx = next((i for i, s in enumerate(sections) if s["heading"] == target), None)

        if operation == "replace":
            if idx is None:
                raise ValueError(f"replace 未找到目标章节: {target!r}")
            sections[idx]["body_lines"] = content.strip().splitlines()
            return self._render_annotation(parsed)

        # append to section
        if idx is None:
            # 目标章节不存在则新建（允许自建 ## 章节）
            sections.append({"heading": target, "body_lines": content.strip().splitlines()})
        else:
            body = list(sections[idx]["body_lines"])
            extra = content.strip().splitlines()
            if body and extra:
                body.append("")
            body.extend(extra)
            sections[idx]["body_lines"] = body
        return self._render_annotation(parsed)

    @staticmethod
    def _is_allowed_extra_heading(heading: str) -> bool:
        return any(heading.startswith(prefix) for prefix in ALLOWED_EXTENSION_SECTION_PREFIXES)

    def _parse_annotation(
        self,
        content: str,
        module_slug: str,
        *,
        ensure_fixed: bool,
    ) -> dict[str, Any]:
        title = module_slug
        sections: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_body: list[str] = []

        raw = content.strip()
        if not raw:
            return {
                "title": module_slug,
                "sections": [{"heading": h, "body_lines": []} for h in FIXED_SECTION_HEADINGS],
            }

        # heading_index: 已出现的章节名 → sections 列表中的下标，用于重复章节合并
        heading_index: dict[str, int] = {}

        def _flush() -> None:
            if current_heading is None:
                return
            if current_heading in heading_index:
                # 重复章节：将内容合并到第一次出现的章节末尾
                existing_section = sections[heading_index[current_heading]]
                extra = [line for line in current_body if line.strip()]
                if extra:
                    if existing_section["body_lines"]:
                        existing_section["body_lines"].append("")
                    existing_section["body_lines"].extend(extra)
            else:
                heading_index[current_heading] = len(sections)
                sections.append(
                    {
                        "heading": current_heading,
                        "body_lines": current_body[:],
                    }
                )

        for line in raw.splitlines():
            if line.startswith("# ") and title == module_slug:
                title = line[2:].strip() or module_slug
                continue
            if line.startswith("## "):
                _flush()
                current_heading = line[3:].strip()
                current_body = []
                continue
            if current_heading is not None:
                current_body.append(line.rstrip())

        _flush()

        if ensure_fixed:
            existing = {s["heading"] for s in sections}
            for h in FIXED_SECTION_HEADINGS:
                if h not in existing:
                    sections.append({"heading": h, "body_lines": []})

        return {"title": title, "sections": sections}

    def _render_annotation(self, parsed: dict[str, Any]) -> str:
        lines = [f"# {parsed['title']}"]
        for s in parsed["sections"]:
            lines.append("")
            lines.append(f"## {s['heading']}")
            lines.extend(s["body_lines"])
        return "\n".join(lines).rstrip() + "\n"

    async def _load_annotation(self, module_slug: str) -> str:
        path = self._annotation_path(module_slug)
        try:
            raw = await _read_file(path)
        except FileNotFoundError:
            raw = ""
        parsed = self._parse_annotation(raw, module_slug, ensure_fixed=True)
        return self._render_annotation(parsed)

    def _annotation_path(self, module_slug: str) -> Path:
        return self._project_root / ".pce" / "annotations" / "modules" / f"{module_slug}.md"


# ---------------------------------------------------------------------------
# run_digest：对外接口
# ---------------------------------------------------------------------------


async def run_digest(
    *,
    project_root: Path,
    serena_client: SerenaClient,
    insight_cache: InsightCache,
    dirty_state: DirtyState,
    model: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Digest 外部入口，供 server._bootstrap 和 handle_sync 调用。

    失败时抛出异常，由调用方捕获并记录 warning；不影响 init/sync 主流程。
    """
    digest_start = time.monotonic()
    # 先 sweep（标记 hash 过时的 insight），cleanup 在 digest 完成后执行
    sweep_start = time.monotonic()
    try:
        await insight_cache.sweep_stale()
    except Exception as e:
        logger.warning("Digest 前 sweep_stale 失败（已忽略）: %s", e)
    finally:
        logger.info("Digest 阶段耗时: sweep_stale=%.2fs", time.monotonic() - sweep_start)

    planner_start = time.monotonic()
    planner = DigestPlanner(project_root=project_root, insight_cache=insight_cache)
    task_list = await planner.build(dirty_state)
    logger.info(
        "Digest 阶段耗时: planner_build=%.2fs (tasks=%d)",
        time.monotonic() - planner_start,
        len(task_list.items),
    )

    if not task_list.items:
        logger.info("Digest 跳过：当前无可执行的模块级 delta")
        seed_start = time.monotonic()
        await _seed_initial_file_baselines_if_missing(project_root=project_root)
        logger.info(
            "Digest 阶段耗时: seed_initial_baselines=%.2fs",
            time.monotonic() - seed_start,
        )
        # 仍然执行 cleanup 清除已 stale 的条目
        cleanup_start = time.monotonic()
        try:
            removed = await insight_cache.cleanup_stale()
            if removed:
                logger.info("Digest cleanup_stale: 删除 %d 条", removed)
        except Exception as e:
            logger.warning("Digest cleanup_stale 失败（已忽略）: %s", e)
        finally:
            logger.info(
                "Digest 阶段耗时: cleanup_stale=%.2fs",
                time.monotonic() - cleanup_start,
            )
        logger.info("Digest 总耗时: %.2fs", time.monotonic() - digest_start)
        return {
            "executed": False,
            "summary": "",
            "resolved_tasks": 0,
            "pending_tasks": 0,
            "deleted_insights": 0,
            "warnings": task_list.warnings,
        }

    task_list_path = project_root.resolve() / _DIGEST_TASKS_REL
    run_tasks_start = time.monotonic()
    try:
        result = await _run_module_tasks(
            task_list=task_list,
            task_list_path=task_list_path,
            project_root=project_root,
            serena_client=serena_client,
            model=model,
            provider=provider,
        )
        logger.info(
            "Digest 阶段耗时: run_module_tasks=%.2fs",
            time.monotonic() - run_tasks_start,
        )
    finally:
        # 无论成功失败，都尝试清理临时任务文件和 stale insight（各自独立兜底，不覆盖原始异常）
        task_file_cleanup_start = time.monotonic()
        try:
            await task_list.delete(task_list_path)
        except Exception as e:
            logger.warning("Digest 清理任务文件失败（已忽略）: %s", e)
        finally:
            logger.info(
                "Digest 阶段耗时: cleanup_task_file=%.2fs",
                time.monotonic() - task_file_cleanup_start,
            )
        cleanup_start = time.monotonic()
        try:
            removed = await insight_cache.cleanup_stale()
            if removed:
                logger.info("Digest cleanup_stale: 删除 %d 条", removed)
        except Exception as e:
            logger.warning("Digest cleanup_stale 失败（已忽略）: %s", e)
        finally:
            logger.info(
                "Digest 阶段耗时: cleanup_stale=%.2fs",
                time.monotonic() - cleanup_start,
            )

    baseline_start = time.monotonic()
    await _advance_file_baselines(project_root=project_root, task_list=task_list)
    await _seed_initial_file_baselines_if_missing(project_root=project_root)
    logger.info(
        "Digest 阶段耗时: advance_and_seed_baselines=%.2fs",
        time.monotonic() - baseline_start,
    )

    # 只删除已被内化（done）的 insight；skipped 说明缺乏证据暂缓处理，保留供后续重试
    delete_insights_start = time.monotonic()
    consumed_ids = _collect_consumed_insight_ids(task_list)
    deleted_count = await insight_cache.delete_by_ids(consumed_ids)
    logger.info(
        "Digest 阶段耗时: delete_consumed_insights=%.2fs (deleted=%d)",
        time.monotonic() - delete_insights_start,
        deleted_count,
    )
    logger.info("Digest 总耗时: %.2fs", time.monotonic() - digest_start)

    return {
        "executed": True,
        "summary": result["summary"],
        "resolved_tasks": result["resolved_tasks"],
        "pending_tasks": result["pending_tasks"],
        "deleted_insights": deleted_count,
        "warnings": result["warnings"],
    }


async def _run_module_tasks(
    *,
    task_list: DigestTaskList,
    task_list_path: Path,
    project_root: Path,
    serena_client: SerenaClient,
    model: str | None,
    provider: str | None,
) -> dict[str, Any]:
    """按模块分批并发执行 DigestAgent，收窄单任务上下文规模。"""
    warnings: list[str] = list(task_list.warnings)
    pending_items = [item for item in task_list.items if item.status == "pending"]
    batch_size = _get_digest_batch_size()
    batches = _chunk_digest_items(pending_items, batch_size=batch_size)

    async def _run_one(index: int, item: DigestTaskItem) -> tuple[int, str, list[str]]:
        sub_task_path = _module_task_list_path(task_list_path, item)
        sub_task_list = DigestTaskList(
            items=[item],
            warnings=[],
            created_at=_utc_now(),
        )
        await sub_task_list.save(sub_task_path)
        try:
            agent = DigestAgent(
                project_root=project_root,
                task_list_path=sub_task_path,
                model=model,
                provider=provider,
            )
            result = await agent.run(task_list=sub_task_list, serena_client=serena_client)
        except Exception as exc:
            return index, "", [f"任务执行失败: {item.id}: {exc}"]
        finally:
            try:
                await sub_task_list.delete(sub_task_path)
            except Exception as exc:
                logger.warning("Digest 清理模块任务文件失败（已忽略）: %s", exc)

        task_warnings = list(result.get("warnings", []))
        if item.status == "pending":
            task_warnings.append(f"任务未完成: {item.id}")
        summary = str(result.get("summary") or "").strip()
        return index, summary, task_warnings

    ordered_summaries: list[str] = [""] * len(pending_items)
    for batch_no, batch in enumerate(batches, start=1):
        logger.info(
            "Digest 批次执行: batch=%d/%d size=%d",
            batch_no,
            len(batches),
            len(batch),
        )
        start_index = (batch_no - 1) * batch_size
        batch_results = await asyncio.gather(
            *[
                _run_one(start_index + offset, item)
                for offset, item in enumerate(batch)
            ],
            return_exceptions=False,
        )
        for index, summary, task_warnings in batch_results:
            ordered_summaries[index] = summary
            warnings.extend(task_warnings)

    summaries = [
        f"[{item.module_slug}] {summary}"
        for item, summary in zip(pending_items, ordered_summaries, strict=False)
        if summary
    ]

    return {
        "summary": "\n\n".join(summaries).strip(),
        "resolved_tasks": sum(1 for item in task_list.items if item.status != "pending"),
        "pending_tasks": len(task_list.pending_items()),
        "warnings": warnings,
    }


def _module_task_list_path(base_path: Path, item: DigestTaskItem) -> Path:
    suffix = item.module_slug or item.id.replace(":", "-")
    return base_path.with_name(f"{base_path.stem}-{suffix}{base_path.suffix}")


async def _advance_file_baselines(
    *,
    project_root: Path,
    task_list: DigestTaskList,
) -> None:
    """仅将已成功消费的模块任务对应文件前移为新的 baseline。"""
    for item in task_list.items:
        if item.status != "done" or item.digest_delta is None:
            continue
        for file_fact in item.digest_delta.changed_files:
            rel_path = str(file_fact.path)
            if file_fact.status == "deleted":
                await delete_file_baseline(rel_path, root_path=project_root)
                continue
            if file_fact.new_content is None or file_fact.new_hash is None:
                continue
            await save_file_baseline(
                rel_path,
                content=file_fact.new_content,
                content_hash=file_fact.new_hash,
                symbols=file_fact.new_symbols,
                root_path=project_root,
            )


async def _seed_initial_file_baselines_if_missing(*, project_root: Path) -> None:
    """若当前不存在任何 baseline，则用现有索引建立初始 baseline。"""
    baselines_dir = project_root / ".pce" / "baselines" / "files"
    if baselines_dir.exists():
        existing = list(baselines_dir.rglob("*.json"))
        if existing:
            return

    snapshot = await load_index(root_path=project_root)
    if snapshot is None:
        return

    for entry in snapshot.entries:
        rel_path = str(entry.file_meta.path)
        abs_path = project_root / rel_path
        try:
            content = await asyncio.to_thread(abs_path.read_text, "utf-8")
        except Exception:
            logger.warning("初始化 baseline 失败，跳过文件: %s", rel_path)
            continue
        await save_file_baseline(
            rel_path,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            symbols=[
                SymbolFact(
                    name=sym.name,
                    kind=sym.kind,
                    line_start=sym.line_start,
                    line_end=sym.line_end,
                )
                for sym in entry.symbols
            ],
            root_path=project_root,
        )


def _collect_consumed_insight_ids(task_list: DigestTaskList) -> list[str]:
    """收集已成功消费模块任务中的 related_insights 条目 ID。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in task_list.items:
        if item.status != "done" or item.digest_delta is None:
            continue
        for insight in item.digest_delta.related_insights:
            if insight.id not in seen:
                seen.add(insight.id)
                result.append(insight.id)
    return result
