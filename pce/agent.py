"""PCE Agent — 手工实现的 ReAct (Reasoning + Acting) 循环。

基于 litellm 的 tool_call 流程驱动 Serena 工具调用,实现:
- 自然语言代码库查询 (pce_query)
- 变更影响边界分析 (pce_impact)

会话管理:
- in-memory 存储 `_sessions: dict[str, list[dict]]`
- 循环以挂钟时间（而非步数）为终止条件
- 上下文接近窗口上限时触发 compact：将已有认知蒸馏为摘要，重建对话窗口后继续
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

# 认知确认回调：接受文件路径列表，执行暂存区标记
AcknowledgeCallback = Callable[[list[str]], Awaitable[None]]

import aiofiles
import litellm

from .models import ImpactResponse, QueryResponse, ReferenceEdge, SymbolRef
from .serena_client import SerenaClient, SerenaClientError

logger = logging.getLogger(__name__)

MAX_SECONDS = 600  # 默认推理时间上限（秒），10 分钟
# 上下文窗口触发 compact 的使用率阈值（预留 20% 用于 compact 操作本身）
_COMPACT_THRESHOLD = 0.80
# step-3.5-flash 的上下文窗口大小（token 数），可通过环境变量覆盖
_CONTEXT_WINDOW = int(os.getenv("PCE_CONTEXT_WINDOW", "256000"))
# litellm 模型名格式: "<provider_prefix>/<model_name>"
# 示例: "step-3.5-flash"            (直接调用 StepFun API)
#        "openrouter/stepfun/step-3.5-flash:free"  (通过 OpenRouter)
#        "anthropic/claude-3-haiku"  (通过 Anthropic)
MODEL = os.getenv("PCE_MODEL", "openrouter/stepfun/step-3.5-flash:free")

SYSTEM_PROMPT_HEADER = """\
你是 PCE (Precision Context Engine) 的核心 Agent。

职责:
1. 使用 Serena MCP 工具检索代码结构、符号定义与引用关系
2. 在有限上下文中给出可靠的代码理解与影响分析结论
3. 严格基于检索证据回答,不编造任何不存在的信息

能力限制:
- 只能使用提供的 Serena 工具和 Memory 中的索引内容
- 不生成实际代码,不执行写操作
- 如果信息不足,明确说明并建议下一步检索方向

**强制流程**:
- 回答任何关于项目代码的问题前,必须先通过 tool_calls 调用 Serena 工具获取证据
- 若已获取的证据不足,继续调用工具,不要凭推测作答
- 完成推理后,必须调用 deliver 工具提交最终结论,禁止直接以文字形式回答

**交付格式指引**:
上层 Agent 可在 query 中明确指定期望格式,PCE Agent 必须严格遵从。未明确指定时,按以下默认规则选择格式:

- 定位类任务(查找函数/类/变量在哪里):
  返回结构化列表,每项包含:
    file: 相对路径(如 pce/agent.py)
    line_range: [start, end]
    name_path: 符号路径(如 PCEAgent/_run_react_loop,遵循 Parent/child 格式)

- 影响分析类任务(pce_impact):
  返回含行号的引用点列表,每项包含:
    file: 相对路径
    line: 引用所在行
    referencing_symbol: 引用该符号的所属函数/方法名
    snippet: 该行附近的代码片段（上下各 2 行）
  并在列表末尾给出建议修改顺序(叶节点优先)

- 问答/理解类任务:
  自然语言回答;凡提及代码位置,必须附上 file:line 格式

输出要求:
- 回答简洁精准,优先给出文件路径和行号定位
- 影响分析时列出所有直接引用点,不要遗漏\
"""

# deliver 是一个虚拟工具,不转发给 Serena,由循环内部拦截处理
# agent 调用它表示主动声明"任务完成",是唯一的正常终止信号
DELIVER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "deliver",
        "description": "提交最终结论并结束当前任务。完成推理后必须调用此工具,不得直接以文字回答。",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "最终结论内容"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "结论置信度",
                },
            },
            "required": ["answer"],
        },
    },
}

# acknowledge_changes 是另一个虚拟工具,用于 PCE Agent 标记已探索并形成正确认知的文件
# Agent 在通过 Serena 工具读取并理解变更文件后调用此工具
ACKNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "acknowledge_changes",
        "description": (
            "标记指定文件的变更已被探索并形成正确认知。"
            "在通过 Serena 工具读取并理解变更文件的最新内容后，"
            "调用此工具更新暂存区状态。可多次调用，渐进式确认。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "已形成认知的文件相对路径列表",
                },
            },
            "required": ["paths"],
        },
    },
}

# 无 tool_calls 时的最大纠正次数(超出后强制终止,视为异常)
_MAX_NO_TOOL_RETRIES = 3
# finish_reason=length 时最大续写次数
_MAX_LENGTH_CONT = 2
# 单次 LLM 调用超时的最大重试次数
_MAX_TIMEOUT_RETRIES = 1


# ============================================================================
# 辅助函数
# ============================================================================


def _safe_json_dumps(value: Any) -> str:
    """安全地将值序列化为 JSON 字符串。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _try_parse_json(value: Any) -> Any:
    """尝试将字符串解析为 JSON,失败时原样返回。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
    return value


def _extract_message(response: Any) -> dict[str, Any]:
    """从 litellm 响应中提取 choice message。"""
    # 处理 Pydantic/dataclass 格式
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])

    if not choices:
        return {"role": "assistant", "content": ""}

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message", {})

    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=False)
        # 防御: model_dump 可能丢失 tool_calls 字段(某些模型返回格式差异)
        if "tool_calls" not in dumped and hasattr(message, "tool_calls"):
            tc = getattr(message, "tool_calls")
            if tc is not None:
                dumped["tool_calls"] = tc
        # 兼容旧版 function_call 格式
        if "function_call" not in dumped and hasattr(message, "function_call"):
            fc = getattr(message, "function_call")
            if fc is not None:
                dumped["function_call"] = fc
        # StepFun 等 provider 要求 assistant 消息的 content 字段必须存在
        if dumped.get("content") is None:
            dumped["content"] = ""
        return dumped
    if isinstance(message, dict):
        # 同上，确保 content 字段存在
        if message.get("content") is None:
            message = {**message, "content": ""}
        return message
    return {"role": "assistant", "content": str(message)}


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """从 message 中提取 tool_calls 列表。"""
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    # 兼容旧版 function_call 格式(某些模型/路由可能使用)
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return [{"id": str(uuid.uuid4()), "function": function_call}]
    # 兼容 function_call 为 Pydantic/dataclass 对象的情况
    if function_call is not None and hasattr(function_call, "name"):
        name = getattr(function_call, "name", None)
        if name:
            return [{
                "id": str(uuid.uuid4()),
                "function": {
                    "name": name,
                    "arguments": getattr(function_call, "arguments", None),
                },
            }]
    return []


def _parse_tool_call_args(raw_args: Any) -> dict[str, Any] | None:
    """解析 tool_call 的 arguments 字段。失败时返回 None（而非空 dict）。

    返回 None 意味着调用方应将其视为解析错误，而非"无参数调用"。
    """
    if raw_args is None:
        return None
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            result = json.loads(raw_args)
            return result if isinstance(result, dict) else None
        except (ValueError, json.JSONDecodeError):
            return None
    return None



# ============================================================================
# 响应解析
# ============================================================================


def _parse_query_response(content: str, session_id: str) -> QueryResponse:
    """将 Agent 输出解析为 QueryResponse。"""
    _FALLBACK_ANSWERS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成推理，请缩小问题范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 推理超出总时长预算，请缩小问题范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，推理异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，推理可能不完整，请重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小问题范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
    }
    if content in _FALLBACK_ANSWERS:
        return QueryResponse(
            answer=_FALLBACK_ANSWERS[content],
            evidence=[],
            related_symbols=[],
            related_files=[],
            session_id=session_id,
        )

    payload = _try_parse_json(content)

    if isinstance(payload, dict):
        answer = str(payload.get("answer") or payload.get("content") or content)

        related_symbols = []
        for item in payload.get("related_symbols") or []:
            if isinstance(item, dict):
                try:
                    related_symbols.append(SymbolRef.model_validate(item))
                except Exception:
                    continue

        related_files = [
            Path(p) for p in (payload.get("related_files") or []) if isinstance(p, str)
        ]
        evidence = [str(e) for e in (payload.get("evidence") or []) if e]

        return QueryResponse(
            answer=answer,
            evidence=evidence,
            related_symbols=related_symbols,
            related_files=related_files,
            session_id=session_id,
        )

    return QueryResponse(
        answer=str(content) if content else "未能生成有效回答",
        evidence=[],
        related_symbols=[],
        related_files=[],
        session_id=session_id,
    )


def _parse_impact_response(content: str, session_id: str) -> ImpactResponse:
    """将 Agent 输出解析为 ImpactResponse。"""
    _FALLBACK_RISKS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成影响分析，请缩小分析范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 分析超出总时长预算，请缩小分析范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，分析异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，分析可能不完整，请重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小分析范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
    }
    if content in _FALLBACK_RISKS:
        return ImpactResponse(
            impact_chain=[],
            boundary=[],
            risks=[_FALLBACK_RISKS[content]],
            unknowns=[],
            session_id=session_id,
        )

    payload = _try_parse_json(content)

    if isinstance(payload, dict):
        impact_chain = []
        for item in payload.get("impact_chain") or []:
            if isinstance(item, dict):
                try:
                    impact_chain.append(ReferenceEdge.model_validate(item))
                except Exception:
                    continue

        boundary = []
        for item in payload.get("boundary") or []:
            if isinstance(item, dict):
                try:
                    boundary.append(SymbolRef.model_validate(item))
                except Exception:
                    continue

        risks = [str(r) for r in (payload.get("risks") or []) if r]
        unknowns = [str(u) for u in (payload.get("unknowns") or []) if u]

        return ImpactResponse(
            impact_chain=impact_chain,
            boundary=boundary,
            risks=risks,
            unknowns=unknowns,
            session_id=session_id,
        )

    # 无法解析为结构化格式,将原始内容作为风险描述
    return ImpactResponse(
        impact_chain=[],
        boundary=[],
        risks=[str(content)] if content else ["分析结果无法解析"],
        unknowns=[],
        session_id=session_id,
    )


# ============================================================================
# 主 Agent 类
# ============================================================================


class PCEAgent:
    """PCE ReAct Agent。

    维护会话状态,驱动 Serena 工具调用,返回推理结论。
    """

    def __init__(
        self,
        model: str | None = None,
        max_seconds: float = MAX_SECONDS,
    ) -> None:
        self._model = model or MODEL
        self._max_seconds = max_seconds
        self._deliver_tool = DELIVER_TOOL
        # in-memory 会话存储,进程重启后丢失(MVP 范围内可接受)
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    async def _read_pce_file(self, path: Path) -> str:
        """读取 .pce 目录下的文件内容,文件不存在时返回占位文本。"""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return f"(未找到 {path.name},索引尚未构建)"
        except Exception as e:
            logger.warning(f"读取文件失败: {path}: {e}")
            return f"(读取 {path.name} 失败)"

    async def _build_system_prompt(self, memory_root: Path | None) -> str:
        """构建 system prompt,注入 structure.md 和 annotations.md 内容。"""
        root = memory_root or Path.cwd()
        pce_dir = root / ".pce"

        structure_md = await self._read_pce_file(pce_dir / "structure.md")
        annotations_md = await self._read_pce_file(pce_dir / "annotations.md")

        return "\n".join([
            SYSTEM_PROMPT_HEADER,
            "",
            "## 项目结构 (structure.md)",
            structure_md.strip(),
            "",
            "## 语义注解 (annotations.md)",
            annotations_md.strip(),
        ])

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, str]:
        """调用 litellm.completion，返回 (response, finish_reason)。

        finish_reason 取值: "stop" | "tool_calls" | "length" | 其他
        缺失时默认 "stop"。
        """
        response = await asyncio.wait_for(
            asyncio.to_thread(
                litellm.completion,
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.2,
            ),
            timeout=60.0,
        )
        # 从 response.choices[0].finish_reason 中提取，兼容 Pydantic 对象和 dict
        finish_reason = "stop"
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])
        if choices:
            choice = choices[0]
            raw = (
                getattr(choice, "finish_reason", None)
                if not isinstance(choice, dict)
                else choice.get("finish_reason")
            )
            if raw:
                finish_reason = str(raw)
        return response, finish_reason

    @staticmethod
    def _extract_next_prompt_size(response: Any) -> int:
        """估算下一轮调用的 prompt 大小，失败时返回 0。

        下一轮 prompt = 本轮 prompt + 本轮 completion（本轮输出会追加到历史）
        因此用 total_tokens（= prompt + completion）作为下一轮窗口占用的预测值，
        无需累加历史（累加会因每轮 prompt 已包含完整历史而产生 O(n²) 重复计数）。
        """
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0

        def _get(attr: str) -> int:
            v = (
                getattr(usage, attr, None)
                if not isinstance(usage, dict)
                else usage.get(attr)
            )
            return int(v) if v else 0

        total = _get("total_tokens")
        if total:
            return total
        # 部分 provider 不返回 total_tokens，退而求其次手动相加
        return _get("prompt_tokens") + _get("completion_tokens")

    async def _maybe_compact(
        self,
        messages: list[dict[str, Any]],
        token_used: int,
        compact_failed: bool = False,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """在上下文使用率达到阈值时触发 compact，蒸馏认知后重建对话窗口。

        Args:
            messages: 当前对话消息列表
            token_used: 上一轮的 total_tokens（= prompt + completion，即下一轮窗口占用预测值）
            compact_failed: 本轮是否已经尝试过 compact 但失败，避免重复尝试

        Returns:
            (新 messages 列表, 更新后的 token_used, compact_failed 标记)
            若未触发 compact，原样返回传入值。
        """
        if (
            _CONTEXT_WINDOW <= 0
            or compact_failed
            or token_used / _CONTEXT_WINDOW < _COMPACT_THRESHOLD
        ):
            return messages, token_used, compact_failed

        logger.info(
            f"上下文使用率 {token_used}/{_CONTEXT_WINDOW} "
            f"({token_used / _CONTEXT_WINDOW:.1%}) 达到阈值，触发 compact"
        )

        # 提取原始用户问题（第一条 user 消息），让摘要不丢失任务目标
        system_msgs = [m for m in messages if m.get("role") == "system"]
        first_user = next(
            (m for m in messages if m.get("role") == "user"), None
        )
        task_hint = (
            f"\n\n当前任务：{first_user['content']}" if first_user else ""
        )

        # 仅传 system + 摘要请求，避免在 token 接近满时再塞满历史
        compact_request: list[dict[str, Any]] = system_msgs + [{
            "role": "user",
            "content": (
                f"请用简洁的自然语言总结目前的推理进展，包括：\n"
                "1. 已确认的关键事实与结论\n"
                "2. 尚未完成的子任务及下一步行动\n"
                "输出应足够精炼，以便在新对话窗口中直接继续推理。"
                f"{task_hint}"
            ),
        }]

        try:
            # 不传工具列表，让模型自由输出摘要文本
            summary_response, _ = await self._completion(compact_request, [])
            summary_msg = _extract_message(summary_response)
            summary_text = str(summary_msg.get("content") or "").strip()
            compact_tokens = self._extract_next_prompt_size(summary_response)
        except Exception as e:
            logger.warning(f"compact 摘要生成失败，跳过压缩（本轮不再重试）: {e}")
            return messages, token_used, True  # 标记失败，本轮不再重试

        if not summary_text:
            logger.warning("compact 摘要为空，跳过压缩（本轮不再重试）")
            return messages, token_used, True

        # 重建消息列表：system prompt + 摘要注入（作为新窗口的起点）
        new_messages: list[dict[str, Any]] = system_msgs + [{
            "role": "user",
            "content": f"[上下文摘要 — 基于前序推理]\n{summary_text}",
        }, {
            "role": "assistant",
            "content": "已了解前序推理摘要，继续执行剩余任务。",
        }]

        logger.info(f"compact 完成，新窗口消息数: {len(new_messages)}")
        # compact 调用本身消耗了 compact_tokens，以此作为新窗口的基线；重置失败标记
        return new_messages, compact_tokens, False

    async def _run_react_loop(
        self,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> str:
        """执行 ReAct 循环直到 agent 调用 deliver 或超出时长预算。

        终止路径:
          正常: agent 调用 deliver(answer=...)
          兜底: __REACT_TIMEOUT_BUDGET__      — 总时长预算耗尽
                __REACT_NO_TOOL_EXHAUSTED__   — 无 tool_calls 纠正次数耗尽
                __REACT_LENGTH_EXHAUSTED__    — length 截断续写次数耗尽
                __REACT_TIMEOUT__             — 单步超时重试次数耗尽
                __REACT_DELIVER_EMPTY__       — deliver 参数为空或解析失败

        计数器独立，互不干扰:
          no_tool_retries      — LLM 无 tool_calls（无论 finish_reason）
          length_continuations — finish_reason=length 触发续写
          timeout_retries      — 每步内独立，asyncio.TimeoutError
        """
        # 构建工具列表：Serena 只读工具 + 虚拟终止工具 + 认知确认工具（可选）
        tools_schema = serena_client.tools_schema + [self._deliver_tool]
        if acknowledge_cb is not None:
            tools_schema = tools_schema + [ACKNOWLEDGE_TOOL]

        no_tool_retries = 0
        length_continuations = 0
        token_used = 0
        compact_failed = False
        start = time.monotonic()

        while True:
            # ── 时间预算检查 ──────────────────────────────────────────────────
            if time.monotonic() - start >= self._max_seconds:
                logger.warning(
                    f"ReAct 循环超出总时长预算 {self._max_seconds}s，强制终止"
                )
                return "__REACT_TIMEOUT_BUDGET__"

            # ── 上下文 compact 检查 ───────────────────────────────────────────
            messages, token_used, compact_failed = await self._maybe_compact(
                messages, token_used, compact_failed
            )

            # ── LLM 调用，内嵌超时重试（每步独立计数）────────────────────────
            timeout_retries = 0
            while True:
                try:
                    response, finish_reason = await self._completion(
                        messages, tools_schema
                    )
                    break
                except asyncio.TimeoutError:
                    timeout_retries += 1
                    if timeout_retries <= _MAX_TIMEOUT_RETRIES:
                        logger.warning(
                            f"模型调用超时(第 {timeout_retries} 次),正在重试当前步骤"
                        )
                        continue
                    logger.warning("模型调用超时，重试次数耗尽，强制终止")
                    return "__REACT_TIMEOUT__"

            token_used = self._extract_next_prompt_size(response)

            message = _extract_message(response)
            if "role" not in message:
                message["role"] = "assistant"

            tool_calls = _extract_tool_calls(message)

            # ── finish_reason=length：输出被截断 ─────────────────────────────
            # 256K 上下文下概率极低，但仍需兜底处理
            if finish_reason == "length":
                length_continuations += 1
                if length_continuations > _MAX_LENGTH_CONT:
                    logger.warning("输出截断续写次数耗尽，强制终止")
                    return "__REACT_LENGTH_EXHAUSTED__"
                if message.get("content"):
                    # 推理文字被截断：保留已有内容，追加"请继续"
                    messages.append(message)
                    messages.append({"role": "user", "content": "请继续"})
                    logger.warning(
                        f"输出被截断(content,第 {length_continuations} 次),追加续写指令"
                    )
                else:
                    # tool_call JSON 被截断：追加空占位 assistant 保持对话结构连续，
                    # 再追加 user 纠正，避免产生"孤立 user 消息"的非法序列
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({
                        "role": "user",
                        "content": "刚才的工具调用格式不完整，请重新完整地调用工具。",
                    })
                    logger.warning(
                        f"输出被截断(tool_call,第 {length_continuations} 次),要求重新调用"
                    )
                continue

            # 正常路径：将 assistant 消息追加到历史
            messages.append(message)

            # ── 无 tool_calls：LLM 违反约束（无论 finish_reason）────────────
            # 统一处理，避免 finish_reason=tool_calls 但实际 tool_calls 为空的漏网情况
            if not tool_calls:
                no_tool_retries += 1
                if no_tool_retries <= _MAX_NO_TOOL_RETRIES:
                    logger.warning(
                        f"无 tool_calls 输出(第 {no_tool_retries} 次),"
                        f"finish_reason={finish_reason},追加强化纠正指令"
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "你刚才直接用文字回答，这违反了强制流程。"
                            "必须先通过 tool_calls 调用工具获取证据，"
                            "完成推理后再用 deliver 提交最终结论，禁止直接回答。"
                        ),
                    })
                else:
                    logger.warning("无 tool_calls 纠正次数耗尽，强制终止循环")
                    return "__REACT_NO_TOOL_EXHAUSTED__"
                continue

            # ── 分拣 deliver、acknowledge 与 Serena 工具调用 ──────────────────
            deliver_call = None
            acknowledge_calls = []
            serena_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                else:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) or getattr(tc, "name", None)
                if name == "deliver":
                    deliver_call = tc
                elif name == "acknowledge_changes":
                    acknowledge_calls.append(tc)
                else:
                    serena_calls.append(tc)

            # 先并行执行所有 Serena 工具（确保工具结果不丢失）
            if serena_calls:
                tool_results = await asyncio.gather(
                    *[self._invoke_tool(tc, serena_client) for tc in serena_calls]
                )
                for result_msg in tool_results:
                    messages.append({"role": "tool", **result_msg})

            # 处理认知确认调用
            for tc in acknowledge_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    tc_id = tc.get("id") or str(uuid.uuid4())
                    raw_args = fn.get("arguments") or tc.get("arguments")
                else:
                    fn = getattr(tc, "function", None)
                    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())
                    raw_args = getattr(fn, "arguments", None)
                args = _parse_tool_call_args(raw_args)
                paths = (args or {}).get("paths", [])
                if acknowledge_cb and paths:
                    try:
                        await acknowledge_cb(paths)
                        ack_result = f"已确认 {len(paths)} 个文件的认知状态"
                    except Exception as e:
                        ack_result = f"认知确认失败: {e}"
                else:
                    ack_result = "无需确认（无变更文件或回调未配置）"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "acknowledge_changes",
                    "content": ack_result,
                })

            # deliver 到来，提取结论并正常终止
            if deliver_call is not None:
                if isinstance(deliver_call, dict):
                    fn = deliver_call.get("function") or {}
                    raw_args = fn.get("arguments") or deliver_call.get("arguments")
                else:
                    fn = getattr(deliver_call, "function", None)
                    raw_args = getattr(fn, "arguments", None)
                args = _parse_tool_call_args(raw_args)
                if args is None:
                    logger.warning("deliver 调用参数解析失败")
                    return "__REACT_DELIVER_EMPTY__"
                answer = args.get("answer")
                if not answer:
                    logger.warning("deliver 调用缺少 answer 参数")
                    return "__REACT_DELIVER_EMPTY__"
                elapsed = time.monotonic() - start
                logger.debug(f"ReAct 循环收到 deliver，正常终止（耗时 {elapsed:.1f}s）")
                return str(answer)

            # 本轮仅有 Serena 工具调用，重置无工具计数，继续下一轮
            no_tool_retries = 0

    async def _invoke_tool(
        self, tool_call: dict[str, Any], serena_client: SerenaClient
    ) -> dict[str, Any]:
        """执行单个工具调用,返回格式化的 tool 消息。"""
        # 提取 tool_call 字段(兼容 dict 和 object 两种格式)
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("id") or str(uuid.uuid4())
            function = tool_call.get("function") or {}
            tool_name = function.get("name") or tool_call.get("name")
            raw_args = function.get("arguments") or tool_call.get("arguments")
        else:
            tool_call_id = getattr(tool_call, "id", None) or str(uuid.uuid4())
            function = getattr(tool_call, "function", {})
            tool_name = getattr(function, "name", None)
            raw_args = getattr(function, "arguments", None)

        if not tool_name:
            return {
                "tool_call_id": tool_call_id,
                "name": "unknown",
                "content": "工具调用缺少 name 字段",
            }

        args = _parse_tool_call_args(raw_args)

        # args 为 None 说明 JSON 解析失败（参数不完整或格式错误）
        # 返回错误 tool result，让 LLM 感知并重新构造正确的调用
        if args is None:
            logger.warning(f"工具参数解析失败: {tool_name}, raw_args={raw_args!r:.100}")
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"工具参数解析失败: 无效的 JSON。请重新调用 {tool_name} 并确保参数格式正确。",
            }

        try:
            result = await serena_client.call(tool_name, args)
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": _safe_json_dumps(result),
            }
        except SerenaClientError as e:
            logger.warning(f"工具调用失败: {tool_name}: {e}")
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"工具调用失败: {e}",
            }

    # ============================================================================
    # 公开接口
    # ============================================================================

    async def query(
        self,
        question: str,
        session_id: str | None = None,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> QueryResponse:
        """执行自然语言查询。

        Args:
            question: 用户问题
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            结构化的查询响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        sid = session_id or str(uuid.uuid4())
        messages = list(self._sessions.get(sid, []))

        # 新会话时注入 system prompt
        if not messages:
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            messages = [{"role": "system", "content": system_content}]

        messages.append({"role": "user", "content": question})
        answer = await self._run_react_loop(messages, serena_client, acknowledge_cb)
        self._sessions[sid] = messages

        return _parse_query_response(answer, sid)

    async def impact(
        self,
        target: str,
        change_type: str,
        session_id: str | None = None,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> ImpactResponse:
        """执行变更影响分析。

        Args:
            target: 分析目标(符号名或文件路径)
            change_type: 变更类型(modify/rename/delete/add_field/change_signature)
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            结构化的影响分析响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        sid = session_id or str(uuid.uuid4())
        messages = list(self._sessions.get(sid, []))

        if not messages:
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            messages = [{"role": "system", "content": system_content}]

        # 构造影响分析专用提示词
        prompt = "\n".join([
            f"请分析对 `{target}` 进行 `{change_type}` 变更的影响边界。",
            "",
            "请使用 Serena 工具查找所有引用,然后以 JSON 格式输出:",
            "{",
            '  "impact_chain": [...],  // 影响链路中的 ReferenceEdge 列表',
            '  "boundary": [...],      // 影响边界的 SymbolRef 列表',
            '  "risks": [...],         // 风险提示字符串列表',
            '  "unknowns": [...]       // 不确定项字符串列表',
            "}",
        ])

        messages.append({"role": "user", "content": prompt})
        answer = await self._run_react_loop(messages, serena_client, acknowledge_cb)
        self._sessions[sid] = messages

        return _parse_impact_response(answer, sid)
