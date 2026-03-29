"""BaseReActAgent — PCE 所有 Agent 的 ReAct 循环骨架。

提取 PCEAgent / digest 特化 Agent 共享的 ReAct 循环、模型降级与工具分拣逻辑，
将子类差异通过钩子方法注入。

公共机制：
- ReAct 主循环（时间预算、轮次管理）
- LLM 调用 + 模型降级链
- finish_reason=length 截断续写
- 无 tool_calls 纠正
- 虚拟工具 vs Serena 工具分拣（虚拟顺序执行，Serena 并发）
- deliver 终止信号
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import litellm
import litellm.exceptions as litellm_exc

from ._env import build_litellm_model, get_completion_overrides, get_env_text
from .prompt_guard import build_prompt_budget, estimate_input_tokens
from .serena_client import SerenaClient, SerenaClientError

logger = logging.getLogger(__name__)

# ============================================================================
# 公共常量
# ============================================================================

DEFAULT_MAX_SECONDS = 600.0
MAX_NO_TOOL_RETRIES = 3
MAX_LENGTH_CONT = 2
MAX_TIMEOUT_RETRIES = 1

# 预算告警参数（子类可通过类属性覆盖）
BUDGET_WARNING_RATIO = 0.25
BUDGET_WARNING_MIN_SECONDS = 45.0
BUDGET_WARNING_MAX_SECONDS = 90.0

# ReAct 异常终止标记
REACT_TIMEOUT_BUDGET = "__REACT_TIMEOUT_BUDGET__"
REACT_NO_TOOL_EXHAUSTED = "__REACT_NO_TOOL_EXHAUSTED__"
REACT_LENGTH_EXHAUSTED = "__REACT_LENGTH_EXHAUSTED__"
REACT_TIMEOUT = "__REACT_TIMEOUT__"
REACT_LLM_EXHAUSTED = "__REACT_LLM_EXHAUSTED__"
REACT_DELIVER_EMPTY = "__REACT_DELIVER_EMPTY__"
REACT_INPUT_TOO_LARGE = "__REACT_INPUT_TOO_LARGE__"

# deliver 虚拟工具 schema（子类可覆盖 _deliver_tool 属性定制参数）
DELIVER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "deliver",
        "description": (
            "提交最终结论并结束当前任务。完成推理后必须调用此工具,不得直接以文字回答。"
            "answer 必须是结构化 Markdown 文本,不得输出 JSON 或 Markdown 代码块。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "最终结论内容，必须是结构化 Markdown 文本",
                },
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


# ============================================================================
# 公共异常
# ============================================================================


class LLMCompletionError(RuntimeError):
    """所有候选模型均失败时抛出，携带完整降级记录。"""

    def __init__(self, attempts: list[dict[str, str]]) -> None:
        self.attempts = attempts
        self.models = [a["model"] for a in attempts]
        parts = [f'{a["model"]}({a["error_type"]}): {a["reason"]}' for a in attempts]
        super().__init__(
            f"LLM fallback chain exhausted; models={self.models}; " + " | ".join(parts)
        )


class PromptBudgetExceeded(RuntimeError):
    """输入上下文在降级后仍超限。"""


# ============================================================================
# 公共辅助函数
# ============================================================================


def _stringify_error(exc: Exception) -> str:
    """提取稳定、可读的异常原因字符串。"""
    return str(getattr(exc, "message", None) or exc or type(exc).__name__)


def _should_fallback_model(exc: Exception) -> bool:
    """判断是否应切换到下一个模型（不可恢复的模型级错误）。"""
    if isinstance(
        exc,
        (
            litellm_exc.RateLimitError,
            litellm_exc.AuthenticationError,
            litellm_exc.PermissionDeniedError,
            litellm_exc.NotFoundError,
        ),
    ):
        return True
    if isinstance(exc, (litellm_exc.BadRequestError, litellm_exc.InvalidRequestError)):
        msg = _stringify_error(exc).lower()
        return any(
            k in msg
            for k in (
                "model not found",
                "unknown model",
                "invalid model",
                "does not exist",
                "no deployments available",
            )
        )
    return False


def _parse_model_fallbacks(raw: str) -> list[str]:
    """解析逗号分隔的 fallback 模型列表，按出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _extract_finish_reason(response: Any) -> str:
    """从 litellm 响应中提取 finish_reason，缺失时默认 'stop'。"""
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
            return str(raw)
    return "stop"


def _safe_json_dumps(value: Any) -> str:
    """安全地将值序列化为 JSON 字符串。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _extract_message(response: Any) -> dict[str, Any]:
    """从 litellm 响应中提取 choice message。"""
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
        if "tool_calls" not in dumped and hasattr(message, "tool_calls"):
            tc = message.tool_calls
            if tc is not None:
                dumped["tool_calls"] = tc
        if "function_call" not in dumped and hasattr(message, "function_call"):
            fc = message.function_call
            if fc is not None:
                dumped["function_call"] = fc
        if dumped.get("content") is None:
            dumped["content"] = ""
        return dumped

    if isinstance(message, dict):
        if message.get("content") is None:
            return {**message, "content": ""}
        return message

    return {"role": "assistant", "content": str(message)}


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """从 message 中提取 tool_calls 列表。"""
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls

    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return [{"id": str(uuid.uuid4()), "function": function_call}]
    if function_call is not None and hasattr(function_call, "name"):
        name = getattr(function_call, "name", None)
        if name:
            return [
                {
                    "id": str(uuid.uuid4()),
                    "function": {
                        "name": name,
                        "arguments": getattr(function_call, "arguments", None),
                    },
                }
            ]
    return []


def _parse_tool_call_args(raw_args: Any) -> dict[str, Any] | None:
    """解析 tool_call 的 arguments 字段。失败时返回 None。"""
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
# ReAct 循环状态
# ============================================================================


@dataclass(slots=True)
class LoopState:
    """ReAct 循环的可变运行状态。

    由 BaseReActAgent.run_loop 创建和管理，子类通过钩子方法访问。
    """

    messages: list[dict[str, Any]]
    start_time: float
    deadline: float
    round_num: int = 0
    no_tool_retries: int = 0
    length_continuations: int = 0
    token_used: int = 0
    budget_warning_used: bool = False
    # 子类可通过 extra 存放专属状态（如 compact_failed、spawn_count 等）
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


# ============================================================================
# deliver 决策
# ============================================================================


@dataclass(slots=True)
class DeliverDecision:
    """on_deliver 钩子的返回值。

    stop=True：循环终止，返回 answer/confidence。
    stop=False：循环继续，extra_messages 追加到对话中（用于追问等场景）。
    """

    stop: bool
    answer: str = ""
    confidence: str | None = None
    extra_messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def finish(cls, answer: str, confidence: str | None = None) -> DeliverDecision:
        return cls(stop=True, answer=answer, confidence=confidence)

    @classmethod
    def continue_with(cls, *messages: dict[str, Any]) -> DeliverDecision:
        return cls(stop=False, extra_messages=list(messages))


# ============================================================================
# BaseReActAgent
# ============================================================================


class BaseReActAgent(ABC):
    """所有 PCE Agent 的 ReAct 循环骨架。

    子类通过实现抽象方法和覆盖钩子来注入特有行为。
    公共循环逻辑在 run_loop() 中实现。
    """

    # 子类可覆盖的类级配置
    _completion_temperature: float = 0.2
    _completion_timeout: float = 60.0
    _enable_budget_warning: bool = False

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        model_fallbacks: list[str] | None = None,
        max_seconds: float = DEFAULT_MAX_SECONDS,
    ) -> None:
        explicit_model = model.strip() if model else None
        explicit_provider = provider.strip() if provider else None

        if explicit_model is None:
            self._provider = get_env_text("PCE_PROVIDER")
            self._model = get_env_text("PCE_MODEL")
            if not self._provider or not self._model:
                raise ValueError("未配置 PCE_PROVIDER / PCE_MODEL，Agent 无法调用模型")
        else:
            self._provider = explicit_provider
            self._model = explicit_model

        raw_fallbacks = (
            model_fallbacks
            if model_fallbacks is not None
            else _parse_model_fallbacks(os.getenv("PCE_MODEL_FALLBACKS", ""))
        )
        self._model_fallbacks = [m for m in raw_fallbacks if m and m != self._model]
        self._max_seconds = float(max_seconds)

    # ── 子类必须实现 ──────────────────────────────────────────────────────────

    @abstractmethod
    def build_tools_schema(
        self,
        serena_client: SerenaClient,
        state: LoopState,
    ) -> list[dict[str, Any]]:
        """构建本轮可用工具 schema（含 deliver + 虚拟工具 + Serena 工具）。"""

    @property
    @abstractmethod
    def virtual_tool_names(self) -> set[str]:
        """返回由子类拦截执行的虚拟工具名集合（不含 deliver）。"""

    @abstractmethod
    async def handle_virtual_tool(
        self,
        tool_call: Any,
        state: LoopState,
    ) -> dict[str, Any] | None:
        """顺序执行单个虚拟工具，返回 tool message dict 或 None。"""

    @abstractmethod
    async def on_deliver(
        self,
        args: dict[str, Any] | None,
        state: LoopState,
    ) -> DeliverDecision:
        """处理 deliver 调用，决定是终止还是继续（如追问未完成任务）。"""

    # ── 子类可选覆盖 ──────────────────────────────────────────────────────────

    async def on_before_round(self, state: LoopState) -> None:
        """每轮 LLM 调用前的钩子。用于 compact、预算检查等。

        若需要提前终止循环，抛出 _LoopEarlyExit。
        """

    async def on_budget_warning(self, state: LoopState) -> None:
        """预算告警钩子。子类可覆盖以注入告警消息到 state.messages。"""

    async def on_before_completion(
        self,
        state: LoopState,
        tools_schema: list[dict[str, Any]],
    ) -> None:
        """在真正调用模型前执行输入预算检查。"""
        self._enforce_input_budget(state, tools_schema)

    def build_no_tool_correction(self, state: LoopState, finish_reason: str) -> dict[str, Any]:
        """无 tool_calls 时的纠正消息。子类可覆盖以定制纠正措辞。"""
        return {
            "role": "user",
            "content": (
                "你刚才直接用文字回答，这违反了强制流程。"
                "必须先通过 tool_calls 调用工具获取证据或推进任务，"
                "完成后再用 deliver 提交最终结论，禁止直接回答。"
            ),
        }

    # ── 公共 ReAct 循环 ──────────────────────────────────────────────────────

    async def run_loop(
        self,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
        *,
        acknowledge_cb: Any | None = None,
        depth: int = 0,
        deadline: float | None = None,
        trace_writer: Any | None = None,
    ) -> tuple[str, str | None]:
        """执行公共 ReAct 循环。

        Args:
            messages:        初始消息历史（会被原地修改）。
            serena_client:   Serena MCP 工具客户端。
            acknowledge_cb:  文件变更确认回调（可选，PCEAgent 使用）。
            depth:           spawn 深度（0=主 Agent，>=1=子 Agent）。
            deadline:        绝对截止时间（monotonic），不传则按 max_seconds 计算。
            trace_writer:    JSONL trace 写入器（可选）。

        Returns:
            (answer, confidence)，正常终止时 confidence 为 deliver 提供的置信度，
            异常终止时 confidence 为 None。
        """
        start = time.monotonic()
        effective_deadline = deadline if deadline is not None else start + self._max_seconds
        budget_seconds = effective_deadline - start

        state = LoopState(
            messages=messages,
            start_time=start,
            deadline=effective_deadline,
        )
        # 预填充 extra：子类钩子通过 state.extra 访问这些运行时状态
        state.extra["serena_client"] = serena_client
        state.extra["acknowledge_cb"] = acknowledge_cb
        state.extra["depth"] = depth
        state.extra["spawn_count"] = 0
        state.extra["compact_failed"] = False
        state.extra["budget_seconds"] = budget_seconds
        if trace_writer is not None:
            state.extra["trace"] = trace_writer

        while True:
            # ── 时间预算检查 ──
            if time.monotonic() >= state.deadline:
                depth = state.extra.get("depth", 0)
                logger.warning(
                    "ReAct 循环超出截止时间，强制终止 (depth=%d)", depth
                )
                return REACT_TIMEOUT_BUDGET, None

            # ── 预算告警 ──
            if self._enable_budget_warning and not state.budget_warning_used:
                budget = state.extra.get("budget_seconds", self._max_seconds)
                threshold = min(
                    BUDGET_WARNING_MAX_SECONDS,
                    max(BUDGET_WARNING_MIN_SECONDS, budget * BUDGET_WARNING_RATIO),
                )
                if state.remaining <= threshold:
                    await self.on_budget_warning(state)
                    state.budget_warning_used = True

            state.round_num += 1

            # ── 子类前置钩子（compact 等） ──
            try:
                await self.on_before_round(state)
            except LLMCompletionError as exc:
                logger.warning("on_before_round 阶段模型降级链耗尽，终止循环: %s", exc)
                return REACT_LLM_EXHAUSTED, None

            # ── 构建工具列表 ──
            tools_schema = self.build_tools_schema(serena_client, state)

            try:
                await self.on_before_completion(state, tools_schema)
            except PromptBudgetExceeded as exc:
                logger.warning("输入上下文超限，终止 ReAct 循环: %s", exc)
                return REACT_INPUT_TOO_LARGE, None

            # ── LLM 调用（含单步超时重试） ──
            timeout_retries = 0
            while True:
                try:
                    response, finish_reason = await self._completion(
                        state.messages, tools_schema
                    )
                    break
                except LLMCompletionError as exc:
                    logger.warning("模型降级链耗尽，终止 ReAct 循环: %s", exc)
                    return REACT_LLM_EXHAUSTED, None
                except TimeoutError:
                    timeout_retries += 1
                    if timeout_retries <= MAX_TIMEOUT_RETRIES:
                        logger.warning(
                            "模型调用超时(第 %d 次)，正在重试", timeout_retries
                        )
                        continue
                    logger.warning("模型调用超时，重试次数耗尽，强制终止")
                    return REACT_TIMEOUT, None

            state.token_used = self._extract_next_prompt_size(response)
            message = _extract_message(response)
            if "role" not in message:
                message["role"] = "assistant"
            tool_calls = _extract_tool_calls(message)

            # ── finish_reason=length：输出被截断 ──
            if finish_reason == "length":
                state.length_continuations += 1
                if state.length_continuations > MAX_LENGTH_CONT:
                    logger.warning("输出截断续写次数耗尽，强制终止")
                    return REACT_LENGTH_EXHAUSTED, None

                if message.get("content"):
                    self._append(state, message)
                    self._append(state, {
                        "role": "user",
                        "content": "输出被截断了，请继续完成剩余内容，并继续通过工具调用推进任务。",
                    })
                else:
                    self._append(state, {"role": "assistant", "content": ""})
                    self._append(state, {
                        "role": "user",
                        "content": "刚才的工具调用格式不完整，请重新完整地调用工具。",
                    })
                continue

            # ── 正常路径：追加 assistant 消息 ──
            self._append(state, message)

            # ── 无 tool_calls：违反约束 ──
            if not tool_calls:
                state.no_tool_retries += 1
                if state.no_tool_retries <= MAX_NO_TOOL_RETRIES:
                    logger.warning(
                        "无 tool_calls 输出(第 %d 次), finish_reason=%s",
                        state.no_tool_retries,
                        finish_reason,
                    )
                    self._append(state, self.build_no_tool_correction(state, finish_reason))
                    continue
                logger.warning("无 tool_calls 纠正次数耗尽，强制终止循环")
                return REACT_NO_TOOL_EXHAUSTED, None

            # 有工具调用则重置 no_tool 计数
            state.no_tool_retries = 0

            # ── 分拣 deliver / 虚拟工具 / Serena 工具 ──
            deliver_call = None
            virtual_calls: list[Any] = []
            serena_calls: list[Any] = []
            vt_names = self.virtual_tool_names

            for tc in tool_calls:
                name = _get_tool_name(tc)
                if name == "deliver":
                    deliver_call = tc
                elif name in vt_names:
                    virtual_calls.append(tc)
                else:
                    serena_calls.append(tc)

            # Serena 工具并发执行
            if serena_calls:
                results = await asyncio.gather(
                    *[self._invoke_serena(tc, serena_client, state=state) for tc in serena_calls]
                )
                for result in results:
                    self._append(state, {"role": "tool", **result})

            # 虚拟工具顺序执行（保证状态更新有序）
            for tc in virtual_calls:
                tool_msg = await self.handle_virtual_tool(tc, state)
                if tool_msg is not None:
                    self._append(state, {"role": "tool", **tool_msg})

            # deliver 处理
            if deliver_call is not None:
                args = _extract_tool_call_args(deliver_call)
                decision = await self.on_deliver(args, state)
                for msg in decision.extra_messages:
                    self._append(state, msg)
                if decision.stop:
                    return decision.answer, decision.confidence

    # ── LLM 调用 + 模型降级链 ────────────────────────────────────────────────

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, str]:
        """调用 litellm.completion，沿降级链尝试所有候选模型。"""
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
                        temperature=self._completion_temperature,
                        **overrides,
                    ),
                    timeout=self._completion_timeout,
                )
                if idx > 0:
                    logger.info("模型降级成功: %s (第 %d 候选)", full_model, idx + 1)
                return response, _extract_finish_reason(response)
            except (TimeoutError, litellm_exc.Timeout) as exc:
                raise TimeoutError("litellm timeout") from exc
            except Exception as exc:
                if not _should_fallback_model(exc):
                    raise
                attempts.append({
                    "model": full_model,
                    "error_type": type(exc).__name__,
                    "reason": _stringify_error(exc),
                })
                if idx < len(model_chain) - 1:
                    logger.warning(
                        "模型调用失败，触发降级: %s -> %s",
                        full_model,
                        model_chain[idx + 1],
                    )
                    continue
                raise LLMCompletionError(attempts) from exc

        raise LLMCompletionError(attempts)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _append(self, state: LoopState, message: dict[str, Any]) -> None:
        """追加消息到对话历史，同时写入 trace（如已配置）。"""
        state.messages.append(message)
        trace = state.extra.get("trace")
        if trace is not None and hasattr(trace, "write"):
            # trace.write 是 async，但我们不等待它完成
            asyncio.ensure_future(trace.write(message))

    def _replace_messages(self, state: LoopState, new_messages: list[dict[str, Any]]) -> None:
        """原地替换对话历史（用于 compact 等场景）。

        直接替换 state.messages 的内容而非重新赋值，保持外部引用有效。
        """
        state.messages.clear()
        state.messages.extend(new_messages)

    def _enforce_input_budget(
        self,
        state: LoopState,
        tools_schema: list[dict[str, Any]],
    ) -> None:
        """统一输入预算检查与默认降级。"""
        budget = build_prompt_budget()
        tokens = estimate_input_tokens(self._model, state.messages, tools_schema)
        if tokens <= budget.soft_input_budget:
            return

        logger.warning(
            "输入上下文接近预算上限: tokens=%d soft=%d hard=%d",
            tokens,
            budget.soft_input_budget,
            budget.hard_input_budget,
        )
        self._prune_history_for_budget(state, tools_schema, budget)
        final_tokens = estimate_input_tokens(self._model, state.messages, tools_schema)
        if final_tokens > budget.hard_input_budget:
            raise PromptBudgetExceeded(
                f"tokens={final_tokens}, hard_budget={budget.hard_input_budget}"
            )

    def _prune_history_for_budget(
        self,
        state: LoopState,
        tools_schema: list[dict[str, Any]],
        budget: Any,
    ) -> None:
        """默认降级策略：裁剪较早的非 system 历史，保留尾部上下文。"""
        messages = list(state.messages)
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(non_system) <= 6:
            return

        note = {
            "role": "user",
            "content": (
                "[系统提示] 为适应上下文预算，较早轮次的对话与工具结果已被折叠。"
                "如缺失证据，请重新调用工具确认。"
            ),
        }
        keep = max(4, min(12, len(non_system)))
        while keep >= 4:
            candidate = [*system_msgs, note, *non_system[-keep:]]
            tokens = estimate_input_tokens(self._model, candidate, tools_schema)
            if tokens <= budget.target_input_budget:
                self._replace_messages(state, candidate)
                return
            keep -= 2

    @staticmethod
    def _extract_next_prompt_size(response: Any) -> int:
        """从 response.usage 中提取下一轮 prompt 预估 token 数。"""
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0

        def _get(attr: str) -> int:
            val = (
                getattr(usage, attr, None)
                if not isinstance(usage, dict)
                else usage.get(attr)
            )
            return int(val) if val else 0

        total = _get("total_tokens")
        if total:
            return total
        return _get("prompt_tokens") + _get("completion_tokens")

    async def _invoke_serena(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
        state: LoopState | None = None,
    ) -> dict[str, Any]:
        """调用 Serena 工具，返回 tool message payload。"""
        tool_call_id = _get_tool_call_id(tool_call)
        tool_name = _get_tool_name(tool_call) or "unknown"
        args = _extract_tool_call_args(tool_call)

        if args is None:
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"工具参数解析失败: 无效的 JSON。请重新调用 {tool_name}。",
            }

        try:
            result = await serena_client.call(tool_name, args)
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": _safe_json_dumps(result),
            }
        except SerenaClientError as exc:
            logger.warning("Serena 工具调用失败: %s: %s", tool_name, exc)
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"Serena 工具调用失败: {exc}",
            }


# ============================================================================
# tool_call 解析辅助（公共静态函数）
# ============================================================================


def _get_tool_name(tool_call: Any) -> str | None:
    """提取 tool_call 的函数名。"""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        return fn.get("name") or tool_call.get("name")
    fn = getattr(tool_call, "function", None)
    return getattr(fn, "name", None) or getattr(tool_call, "name", None)


def _get_tool_call_id(tool_call: Any) -> str:
    """提取 tool_call 的 ID。"""
    if isinstance(tool_call, dict):
        return str(tool_call.get("id") or uuid.uuid4())
    return str(getattr(tool_call, "id", None) or uuid.uuid4())


def _extract_tool_call_args(tool_call: Any) -> dict[str, Any] | None:
    """提取并解析 tool_call 的 arguments。"""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function") or {}
        raw = fn.get("arguments") or tool_call.get("arguments")
    else:
        fn = getattr(tool_call, "function", None)
        raw = getattr(fn, "arguments", None)
    return _parse_tool_call_args(raw)
