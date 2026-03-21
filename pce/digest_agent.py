"""DigestAgent — 认知整合 Agent。

在 pce_init / pce_sync 完成 Serena 索引重建后，将 InsightCache 积累的细粒度观察
和 dirty_files 带来的代码变更，内化到 `.pce/annotations/modules/*.md` 中。

设计原则：
- 一次完整 ReAct 循环，Agent 自主决定探索和更新顺序
- 任务清单（DigestTaskList）注入 system prompt，作为无遗漏保障，不约束执行顺序
- 虚拟工具（任务追踪 + annotation 读写）由 Python 拦截，不透传给 Serena
- 失败为 best-effort：异常只记 warning，不影响 init/sync 主流程
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import litellm.exceptions as litellm_exc

from ._env import build_litellm_model, get_completion_overrides, get_env_text
from .agent import (
    LLMCompletionError,
    _extract_finish_reason,
    _extract_message,
    _extract_tool_calls,
    _parse_tool_call_args,
    _safe_json_dumps,
    _should_fallback_model,
    _stringify_error,
)
from .insight_cache import InsightCache
from .serena_client import SerenaClient, SerenaClientError
from .staging import DirtyState

logger = logging.getLogger(__name__)

# annotation 固定章节，不可删除
FIXED_SECTION_HEADINGS: tuple[str, ...] = (
    "覆盖文件",
    "核心职责",
    "关键符号",
    "关键流程",
    "外部协作",
    "风险与约束",
)

_DIGEST_TASKS_REL = Path(".pce") / "digest_tasks.json"
_DEFAULT_MAX_SECONDS = 300.0
_MAX_NO_TOOL_RETRIES = 3
_MAX_TIMEOUT_RETRIES = 1
_MAX_LENGTH_CONT = 2


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _parse_model_fallbacks(raw: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in raw.split(","):
        m = item.strip()
        if m and m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# 任务数据结构
# ---------------------------------------------------------------------------


@dataclass
class DigestTaskItem:
    id: str
    kind: str             # "insight" | "dirty_module" | "dirty_file"
    status: str           # "pending" | "done" | "skipped"
    note: str | None = None
    # insight 类型字段
    insight_id: str | None = None
    scope: str | None = None
    insight_summary: str | None = None
    temporal_stale: bool = False
    # 模块/文件变更类型字段
    module_slug: str | None = None
    dirty_files: list[str] = field(default_factory=list)


@dataclass
class DigestTaskList:
    items: list[DigestTaskItem]
    warnings: list[str]
    created_at: datetime

    def all_resolved(self) -> bool:
        return not any(item.status == "pending" for item in self.items)

    def pending_items(self) -> list[DigestTaskItem]:
        return [item for item in self.items if item.status == "pending"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "warnings": list(self.warnings),
            "items": [asdict(item) for item in self.items],
        }

    async def save(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        await _write_file_atomic(path, payload)

    async def delete(self, path: Path) -> None:
        await asyncio.to_thread(path.unlink, True)


# ---------------------------------------------------------------------------
# DigestPlanner：构建任务清单
# ---------------------------------------------------------------------------


class DigestPlanner:
    """基于 annotations/index.md + InsightCache + DirtyState 构建 DigestTaskList。"""

    def __init__(self, project_root: Path, insight_cache: InsightCache) -> None:
        self._project_root = project_root.resolve()
        self._insight_cache = insight_cache

    async def build(self, dirty_state: DirtyState) -> DigestTaskList:
        file_to_module, warnings = await self._load_file_module_map()
        dirty_paths = list(dict.fromkeys(dirty_state.changed + dirty_state.deleted))

        # 按模块归并 dirty files
        module_dirty: dict[str, list[str]] = {}
        orphan_dirty: list[str] = []
        for path in dirty_paths:
            slug = file_to_module.get(path)
            if slug is None:
                orphan_dirty.append(path)
                warnings.append(f"dirty 文件未命中 annotation 模块映射，Agent 自行判断归属: {path}")
            else:
                module_dirty.setdefault(slug, []).append(path)

        items: list[DigestTaskItem] = []
        records = await self._insight_cache.get_all_records(include_stale=True)
        represented_modules: set[str] = set()

        # insight 任务（含时序冲突标注）
        for record in records:
            slug = file_to_module.get(record.scope)
            related_dirty = module_dirty.get(slug, []) if slug else []
            # temporal_stale: insight 采集时间早于 dirty 或本身已 stale
            temporal_stale = record.stale or bool(related_dirty)

            if slug is None:
                warnings.append(
                    f"insight scope 未命中模块映射，Agent 自行决定如何处理: {record.scope}"
                )
            else:
                represented_modules.add(slug)

            summary = await self._insight_cache.get_entry_content(record.id)
            items.append(
                DigestTaskItem(
                    id=f"insight:{record.id}",
                    kind="insight",
                    status="pending",
                    insight_id=record.id,
                    scope=record.scope,
                    insight_summary=summary,
                    temporal_stale=temporal_stale,
                    module_slug=slug,
                    dirty_files=list(related_dirty),
                )
            )

        # dirty_module 任务（无对应 insight 的模块变更）
        for slug, files in sorted(module_dirty.items()):
            if slug in represented_modules:
                continue
            items.append(
                DigestTaskItem(
                    id=f"dirty:{slug}",
                    kind="dirty_module",
                    status="pending",
                    temporal_stale=True,
                    module_slug=slug,
                    dirty_files=list(files),
                )
            )

        # dirty_file 任务（未命中任何模块的文件，让 Agent 自主决定）
        for path in orphan_dirty:
            items.append(
                DigestTaskItem(
                    id=f"dirty-file:{path}",
                    kind="dirty_file",
                    status="pending",
                    scope=path,
                    temporal_stale=True,
                    module_slug=None,
                    dirty_files=[path],
                )
            )

        return DigestTaskList(items=items, warnings=warnings, created_at=_utc_now())

    async def _load_file_module_map(self) -> tuple[dict[str, str], list[str]]:
        """解析 annotations/index.md，返回 {文件路径: module_slug} 映射。"""
        index_path = self._project_root / ".pce" / "annotations" / "index.md"
        warnings: list[str] = []
        try:
            raw = await _read_file(index_path)
        except FileNotFoundError:
            return {}, ["未找到 .pce/annotations/index.md，DigestAgent 只能依赖自身探索"]
        except Exception as e:
            return {}, [f"读取 annotations/index.md 失败: {e}"]

        mapping: dict[str, str] = {}
        current_name: str | None = None
        current_slug: str | None = None
        current_files: list[str] = []

        def flush() -> None:
            if current_slug is None:
                return
            for fp in current_files:
                if fp in mapping and mapping[fp] != current_slug:
                    warnings.append(
                        f"文件归属到多个模块，取最后一次映射: {fp} -> {current_slug}"
                    )
                mapping[fp] = current_slug

        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                flush()
                current_name = stripped[3:].strip()
                current_slug = _module_slug_from_name(current_name)
                current_files = []
                continue
            if current_name is None:
                continue
            # 解析 "文件：a.py, b.py"
            if stripped.startswith("文件：") or stripped.startswith("文件:"):
                payload = re.split(r"[：:]", stripped, maxsplit=1)[1]
                current_files = [p.strip() for p in payload.split(",") if p.strip()]
                continue
            # 从 "详细认知：.pce/annotations/modules/xxx.md" 中提取真实 slug
            if stripped.startswith("详细认知：") or stripped.startswith("详细认知:"):
                match = re.search(r"modules/([^/\s]+)\.md", stripped)
                if match:
                    current_slug = match.group(1).strip()

        flush()
        return mapping, warnings


# ---------------------------------------------------------------------------
# DigestAgent：ReAct 循环执行认知整合
# ---------------------------------------------------------------------------


class DigestAgent:
    """认知整合专用 ReAct Agent。

    复用 PCEAgent 的模型调用/降级机制，但工具集完全不同：
    - 虚拟工具（任务追踪 + annotation 读写）由 Python 拦截
    - Serena 只读工具透传，供 Agent 按需探索代码库
    """

    def __init__(
        self,
        project_root: Path,
        task_list_path: Path,
        model: str | None = None,
        provider: str | None = None,
        model_fallbacks: list[str] | None = None,
        max_seconds: float = _DEFAULT_MAX_SECONDS,
    ) -> None:
        explicit_model = model.strip() if model else None
        explicit_provider = provider.strip() if provider else None

        if explicit_model is None:
            self._provider = get_env_text("PCE_PROVIDER")
            self._model = get_env_text("PCE_MODEL")
            if not self._provider or not self._model:
                raise ValueError(
                    "未配置 PCE_PROVIDER / PCE_MODEL，DigestAgent 无法调用模型"
                )
        else:
            self._provider = explicit_provider
            self._model = explicit_model

        raw_fallbacks = (
            model_fallbacks
            if model_fallbacks is not None
            else _parse_model_fallbacks(os.getenv("PCE_MODEL_FALLBACKS", ""))
        )
        self._model_fallbacks = [m for m in raw_fallbacks if m and m != self._model]
        self._project_root = project_root.resolve()
        self._task_list_path = task_list_path
        self._max_seconds = max_seconds
        self._virtual_tools = self._build_virtual_tools()
        self._virtual_tool_names = {
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

        summary = await self._run_react_loop(messages, serena_client)
        resolved = sum(1 for item in task_list.items if item.status != "pending")
        return {
            "summary": summary,
            "resolved_tasks": resolved,
            "pending_tasks": len(task_list.pending_items()),
            "warnings": list(task_list.warnings),
        }

    def _build_system_prompt(self, task_list: DigestTaskList) -> str:
        task_json = json.dumps(task_list.to_dict(), ensure_ascii=False, indent=2)
        fixed = "、".join(FIXED_SECTION_HEADINGS)
        return "\n".join(
            [
                "你是 PCE 的 DigestAgent，负责把历史 insight 和代码变更内化到模块 annotation 文档。",
                "",
                "## 目标",
                "1. 读取任务清单，理解每个待内化项（insight 或 dirty_file）。",
                "2. 按需通过 Serena 工具探索代码获取最新证据。",
                "3. 用 read_annotation + write_annotation_patch 更新对应模块的 annotation。",
                "4. 完成或跳过每个任务后，调用 mark_task_done / mark_task_skipped 记录结论。",
                "5. 所有任务处理完毕后，调用 deliver(summary=...) 结束。",
                "",
                "## 固定约束",
                f"- annotation 固定章节不可删除：{fixed}",
                "- 可新增额外 `##` 章节（例如 `## Prompt 设计细节`）。",
                "- `replace` 按 `##` 级标题定位，替换整个章节正文（包含想保留的内容）。",
                "- `rewrite` 提交完整 Markdown，必须显式保留全部 6 个固定章节。",
                "- `append` 在指定章节末尾追加内容；target=null 时追加到文档末尾。",
                "- 每个任务必须有明确结论（done 或 skipped），不允许静默跳过。",
                "- mark_task_done / mark_task_skipped 的 note 字段必须填写。",
                "",
                "## 时序说明",
                "- `temporal_stale=true` 表示该 insight 的采集时间早于文件变更，",
                "  insight 内容可能已经过时，必须结合 Serena 工具验证最新代码再更新。",
                "- `dirty_files` 是与该任务关联的最新改动文件，应优先核对。",
                "",
                "## 当前任务清单",
                task_json,
            ]
        )

    def _build_virtual_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_task_list",
                    "description": "读取当前 digest 任务清单（含任务状态）。",
                    "parameters": {"type": "object", "properties": {}},
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
            },
        ]

    async def _run_react_loop(
        self,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
    ) -> str:
        tools_schema = serena_client.tools_schema + self._virtual_tools
        no_tool_retries = 0
        length_continuations = 0
        deliver_guard_used = False
        start = time.monotonic()

        while True:
            if time.monotonic() - start >= self._max_seconds:
                raise RuntimeError("DigestAgent ReAct 循环超时，已达最大时间预算")

            # 模型调用（含单步超时重试）
            timeout_retries = 0
            while True:
                try:
                    response, finish_reason = await self._completion(messages, tools_schema)
                    break
                except LLMCompletionError:
                    raise
                except asyncio.TimeoutError:
                    timeout_retries += 1
                    if timeout_retries <= _MAX_TIMEOUT_RETRIES:
                        logger.warning("DigestAgent 模型调用超时，正在重试")
                        continue
                    raise RuntimeError("DigestAgent 模型调用超时，重试次数耗尽")

            message = _extract_message(response)
            if "role" not in message:
                message["role"] = "assistant"
            messages.append(message)

            # 输出被截断，续写
            if finish_reason == "length":
                length_continuations += 1
                if length_continuations > _MAX_LENGTH_CONT:
                    raise RuntimeError("DigestAgent 输出连续被截断，已放弃本轮")
                messages.append({
                    "role": "user",
                    "content": "输出被截断了，请继续完成剩余内容，并继续用工具调用处理任务。",
                })
                continue

            tool_calls = _extract_tool_calls(message)
            if not tool_calls:
                no_tool_retries += 1
                if no_tool_retries <= _MAX_NO_TOOL_RETRIES:
                    messages.append({
                        "role": "user",
                        "content": (
                            "你刚才没有调用工具。DigestAgent 必须通过工具处理任务，"
                            "请用 read_task_list 查看任务，再继续操作。"
                        ),
                    })
                    continue
                raise RuntimeError("DigestAgent 连续未产生 tool_calls，已终止")

            no_tool_retries = 0

            # 分拣：虚拟工具 vs Serena 工具
            virtual_calls: list[Any] = []
            serena_calls: list[Any] = []
            deliver_summary: str | None = None

            for tc in tool_calls:
                name = self._get_tool_name(tc)
                if name in self._virtual_tool_names:
                    virtual_calls.append(tc)
                else:
                    serena_calls.append(tc)

            # Serena 工具并发执行
            if serena_calls:
                serena_results = await asyncio.gather(
                    *[self._invoke_serena_tool(tc, serena_client) for tc in serena_calls]
                )
                for res in serena_results:
                    messages.append({"role": "tool", **res})

            # 虚拟工具顺序执行（保证任务状态更新有序）
            for tc in virtual_calls:
                tool_msg, maybe_summary = await self._handle_virtual_tool(tc)
                if tool_msg is not None:
                    messages.append({"role": "tool", **tool_msg})
                if maybe_summary is not None:
                    deliver_summary = maybe_summary

            # deliver 处理
            if deliver_summary is not None:
                assert self._task_list is not None
                if not self._task_list.all_resolved() and not deliver_guard_used:
                    pending = [
                        {
                            "id": item.id,
                            "kind": item.kind,
                            "module_slug": item.module_slug,
                            "scope": item.scope,
                            "dirty_files": item.dirty_files,
                        }
                        for item in self._task_list.pending_items()
                    ]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"还有 {len(pending)} 个未解决的任务，请继续处理后再 deliver：\n"
                            f"{json.dumps(pending, ensure_ascii=False, indent=2)}"
                        ),
                    })
                    deliver_guard_used = True
                    continue
                if not self._task_list.all_resolved():
                    # 追问后仍有遗漏，记录 warning 但允许退出
                    for item in self._task_list.pending_items():
                        self._task_list.warnings.append(
                            f"任务未被 Agent 处理即结束: {item.id} ({item.kind})"
                        )
                return deliver_summary

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, str]:
        overrides = get_completion_overrides()
        model_chain = [
            build_litellm_model(self._provider, m)
            for m in [self._model, *self._model_fallbacks]
        ]
        attempts: list[dict[str, str]] = []

        for idx, full_model in enumerate(model_chain):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        litellm.completion,
                        model=full_model,
                        messages=messages,
                        tools=tools if tools else None,
                        temperature=0.1,
                        **overrides,
                    ),
                    timeout=60.0,
                )
                if idx > 0:
                    logger.info("DigestAgent 模型降级成功: %s", full_model)
                return response, _extract_finish_reason(response)
            except (asyncio.TimeoutError, litellm_exc.Timeout):
                raise asyncio.TimeoutError("litellm timeout")
            except Exception as exc:
                if not _should_fallback_model(exc):
                    raise
                attempts.append({
                    "model": full_model,
                    "error_type": type(exc).__name__,
                    "reason": _stringify_error(exc),
                })
                if idx < len(model_chain) - 1:
                    logger.warning("DigestAgent 模型调用失败，尝试降级: %s", full_model)
                    continue
                raise LLMCompletionError(attempts) from exc

        raise LLMCompletionError(attempts)

    @staticmethod
    def _get_tool_name(tool_call: Any) -> str | None:
        if isinstance(tool_call, dict):
            fn = tool_call.get("function") or {}
            return fn.get("name") or tool_call.get("name")
        fn = getattr(tool_call, "function", None)
        return getattr(fn, "name", None) or getattr(tool_call, "name", None)

    @staticmethod
    def _get_tool_call_id(tool_call: Any) -> str:
        if isinstance(tool_call, dict):
            return str(tool_call.get("id") or uuid.uuid4())
        return str(getattr(tool_call, "id", None) or uuid.uuid4())

    async def _invoke_serena_tool(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
    ) -> dict[str, Any]:
        tc_id = self._get_tool_call_id(tool_call)
        name = self._get_tool_name(tool_call) or "unknown"
        raw_args = (
            tool_call.get("function", {}).get("arguments")
            if isinstance(tool_call, dict)
            else getattr(getattr(tool_call, "function", None), "arguments", None)
        )
        args = _parse_tool_call_args(raw_args)
        if args is None:
            return {
                "tool_call_id": tc_id,
                "name": name,
                "content": f"工具参数解析失败: {name}",
            }
        try:
            result = await serena_client.call(name, args)
            return {
                "tool_call_id": tc_id,
                "name": name,
                "content": _safe_json_dumps(result),
            }
        except SerenaClientError as e:
            return {
                "tool_call_id": tc_id,
                "name": name,
                "content": f"Serena 工具调用失败: {e}",
            }

    async def _handle_virtual_tool(
        self,
        tool_call: Any,
    ) -> tuple[dict[str, Any] | None, str | None]:
        assert self._task_list is not None

        tc_id = self._get_tool_call_id(tool_call)
        name = self._get_tool_name(tool_call) or "unknown"
        raw_args = (
            tool_call.get("function", {}).get("arguments")
            if isinstance(tool_call, dict)
            else getattr(getattr(tool_call, "function", None), "arguments", None)
        )
        args: dict[str, Any] = _parse_tool_call_args(raw_args) or {}

        def _err(msg: str) -> tuple[dict[str, Any], None]:
            return {"tool_call_id": tc_id, "name": name, "content": msg}, None

        def _ok(msg: str) -> tuple[dict[str, Any], None]:
            return {"tool_call_id": tc_id, "name": name, "content": msg}, None

        if name == "deliver":
            summary = str(args.get("summary") or "").strip()
            if not summary:
                return _err("deliver.summary 不能为空")
            return None, summary

        if name == "read_task_list":
            return _ok(_safe_json_dumps(self._task_list.to_dict()))

        if name in {"mark_task_done", "mark_task_skipped"}:
            task_id = str(args.get("task_id") or "").strip()
            note = str(args.get("note") or "").strip()
            if not task_id:
                return _err("task_id 不能为空")
            if not note:
                return _err("note 不能为空")
            task = next((item for item in self._task_list.items if item.id == task_id), None)
            if task is None:
                return _err(f"未找到任务: {task_id}")
            task.status = "done" if name == "mark_task_done" else "skipped"
            task.note = note
            await self._task_list.save(self._task_list_path)
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
            target_str = str(target).strip() if isinstance(target, str) and target.strip() else None
            content = str(args.get("content") or "")
            if not module_slug:
                return _err("module_slug 不能为空")
            try:
                original = await self._load_annotation(module_slug)
                updated = self._apply_patch(original, operation, target_str, content, module_slug)
                await _write_file_atomic(self._annotation_path(module_slug), updated)
                return _ok(
                    f"annotation 已更新: module={module_slug}, operation={operation}, "
                    f"target={target_str!r}"
                )
            except Exception as e:
                return _err(f"annotation 更新失败: {e}")

        return _err(f"未知虚拟工具: {name}")

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
            # 校验 6 个固定章节存在
            parsed = self._parse_annotation(content, module_slug, ensure_fixed=False)
            headings = {s["heading"] for s in parsed["sections"]}
            missing = [h for h in FIXED_SECTION_HEADINGS if h not in headings]
            if missing:
                raise ValueError(f"rewrite 缺少固定章节: {missing}")
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
                "sections": [
                    {"heading": h, "body_lines": []} for h in FIXED_SECTION_HEADINGS
                ],
            }

        for line in raw.splitlines():
            if line.startswith("# ") and title == module_slug:
                title = line[2:].strip() or module_slug
                continue
            if line.startswith("## "):
                if current_heading is not None:
                    sections.append({
                        "heading": current_heading,
                        "body_lines": current_body[:],
                    })
                current_heading = line[3:].strip()
                current_body = []
                continue
            if current_heading is not None:
                current_body.append(line.rstrip())

        if current_heading is not None:
            sections.append({"heading": current_heading, "body_lines": current_body[:]})

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
        return (
            self._project_root / ".pce" / "annotations" / "modules" / f"{module_slug}.md"
        )


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
    # 先 sweep（标记 hash 过时的 insight），cleanup 在 digest 完成后执行
    try:
        await insight_cache.sweep_stale()
    except Exception as e:
        logger.warning("Digest 前 sweep_stale 失败（已忽略）: %s", e)

    planner = DigestPlanner(project_root=project_root, insight_cache=insight_cache)
    task_list = await planner.build(dirty_state)

    if not task_list.items:
        logger.info("Digest 跳过：无 insight 也无 dirty_files")
        # 仍然执行 cleanup 清除已 stale 的条目
        try:
            removed = await insight_cache.cleanup_stale()
            if removed:
                logger.info("Digest cleanup_stale: 删除 %d 条", removed)
        except Exception as e:
            logger.warning("Digest cleanup_stale 失败（已忽略）: %s", e)
        return {
            "executed": False,
            "summary": "",
            "resolved_tasks": 0,
            "pending_tasks": 0,
            "deleted_insights": 0,
            "warnings": task_list.warnings,
        }

    task_list_path = project_root.resolve() / _DIGEST_TASKS_REL
    try:
        await task_list.save(task_list_path)

        agent = DigestAgent(
            project_root=project_root,
            task_list_path=task_list_path,
            model=model,
            provider=provider,
        )

        result = await agent.run(task_list=task_list, serena_client=serena_client)
    finally:
        # 无论成功失败，都尝试清理临时任务文件和 stale insight（各自独立兜底，不覆盖原始异常）
        try:
            await task_list.delete(task_list_path)
        except Exception as e:
            logger.warning("Digest 清理任务文件失败（已忽略）: %s", e)
        try:
            removed = await insight_cache.cleanup_stale()
            if removed:
                logger.info("Digest cleanup_stale: 删除 %d 条", removed)
        except Exception as e:
            logger.warning("Digest cleanup_stale 失败（已忽略）: %s", e)

    # 只删除已被内化（done）的 insight；skipped 说明缺乏证据暂缓处理，保留供后续重试
    consumed_ids = [
        item.insight_id
        for item in task_list.items
        if item.insight_id and item.status == "done"
    ]
    deleted_count = await insight_cache.delete_by_ids(consumed_ids)

    return {
        "executed": True,
        "summary": result["summary"],
        "resolved_tasks": result["resolved_tasks"],
        "pending_tasks": result["pending_tasks"],
        "deleted_insights": deleted_count,
        "warnings": result["warnings"],
    }
