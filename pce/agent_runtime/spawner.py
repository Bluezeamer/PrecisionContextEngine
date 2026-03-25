"""SubAgent spawn 执行器。

职责：
- 验证 spawn 调用的合法性（深度、预算、参数）
- 构建子 Agent 消息历史并调用 run_loop
- 将结果封装为 SpawnResult（失败不向外抛异常）

设计说明：
- `run_loop_fn` 回调注入，避免直接依赖 PCEAgent，消除循环导入。
- 预算策略：`remaining < MIN_SPAWN_BUDGET` 时拒绝；否则将 requested 向下 clamp
  到 `min(requested, remaining * MAX_SPAWN_BUDGET_RATIO)`，让子 Agent 尽力而为。
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from .contracts import (
    MAX_SPAWN_BUDGET_RATIO,
    MIN_SPAWN_BUDGET,
    SpawnErrorCode,
    SpawnRequest,
    SpawnResult,
    SpawnStatus,
)

logger = logging.getLogger(__name__)

# 回调签名：与 PCEAgent.run_loop 的关键字参数对齐
RunLoopFn = Callable[..., Awaitable[tuple[str, str | None]]]

# 子 Agent system prompt（简洁版，无项目结构注入）
_CHILD_SYSTEM_PROMPT = """\
你是 PCE SubAgent，只负责完成一个已定义的子任务，不参与主线决策。

规则：
1. 先用 Serena 工具取证，再调用 deliver 提交结论；禁止猜测，禁止调用 spawn_agent。
2. 只回答当前子任务：不要扩展到主问题，不要补充与任务无关的背景。
3. deliver.answer 精炼为"核心结论 + 关键证据定位（file:line 或符号路径）"，避免冗长解释。
4. 若证据不足，在 answer 中明确说明"缺失信息及建议的下一步检索动作"，再 deliver。
5. confidence 仅反映本子任务的证据充分性，不代表主线最终结论的可靠性。

输出以可被主 Agent 直接整合为目标：简短、结构化、少修辞。\
"""


def _make_failed(
    task_id: str,
    elapsed: float,
    code: SpawnErrorCode,
    message: str,
) -> SpawnResult:
    logger.warning("spawn_agent 失败 [%s] task_id=%s: %s", code.value, task_id, message)
    return SpawnResult(
        ok=False,
        task_id=task_id,
        status=SpawnStatus.FAILED,
        answer="",
        confidence="low",
        elapsed_seconds=elapsed,
        error_code=code,
        error_message=message,
    )


async def invoke_spawn(
    request: SpawnRequest,
    serena_client: Any,  # ToolProvider Protocol，避免导入 SerenaClient 产生循环依赖
    parent_depth: int,
    parent_deadline: float,
    run_loop_fn: RunLoopFn,
) -> SpawnResult:
    """执行一次 SubAgent 调用。

    Args:
        request:        spawn 参数（已通过基础校验）。
        serena_client:  工具层（ToolProvider Protocol），与父 Agent 共享。
        parent_depth:   父 Agent 的 spawn 深度（0 = 主 Agent，>=1 = 子 Agent）。
        parent_deadline: 父 Agent 的绝对截止时间（monotonic）。
        run_loop_fn:    注入的 ReAct 循环函数（PCEAgent.run_loop 的绑定方法）。

    Returns:
        SpawnResult，失败时 ok=False，不抛出异常。
    """
    started = time.monotonic()
    task_id = str(uuid.uuid4())

    try:
        # ── 深度检查 ────────────────────────────────────────────────────────────
        if parent_depth >= 1:
            return _make_failed(
                task_id=task_id,
                elapsed=time.monotonic() - started,
                code=SpawnErrorCode.SUBAGENT_DEPTH_EXCEEDED,
                message="超过最大 spawn 深度限制（max_depth=1），禁止递归 spawn",
            )

        # ── 预算检查与裁剪 ──────────────────────────────────────────────────────
        remaining = parent_deadline - time.monotonic()
        if remaining < MIN_SPAWN_BUDGET:
            return _make_failed(
                task_id=task_id,
                elapsed=time.monotonic() - started,
                code=SpawnErrorCode.SUBAGENT_BUDGET_REJECTED,
                message=(
                    f"父剩余预算不足最小子预算，"
                    f"remaining={remaining:.2f}s, min_required={MIN_SPAWN_BUDGET:.2f}s"
                ),
            )

        # 预算策略：
        # 1. max_allowed < MIN_SPAWN_BUDGET → 父剩余不足，直接拒绝
        # 2. 否则：将 requested 向下 clamp 到 max_allowed（不向上抬高），静默缩减
        max_allowed = remaining * MAX_SPAWN_BUDGET_RATIO
        if max_allowed < MIN_SPAWN_BUDGET:
            return _make_failed(
                task_id=task_id,
                elapsed=time.monotonic() - started,
                code=SpawnErrorCode.SUBAGENT_BUDGET_REJECTED,
                message=(
                    f"父剩余预算比例上限不足最小子预算，"
                    f"remaining={remaining:.2f}s, max_allowed={max_allowed:.2f}s, "
                    f"min_required={MIN_SPAWN_BUDGET:.2f}s"
                ),
            )

        # 只向下裁剪，不向上抬高（保证不突破比例上限）
        child_seconds = min(float(request.allocated_seconds), max_allowed)
        if child_seconds < request.allocated_seconds:
            logger.info(
                "spawn_agent 预算裁剪: requested=%.1fs → actual=%.1fs (remaining=%.1fs)",
                request.allocated_seconds,
                child_seconds,
                remaining,
            )

        # ── 构建子 Agent 消息历史 ───────────────────────────────────────────────
        context_str = ""
        if request.context:
            import json
            context_str = f"\n\n补充上下文：\n{json.dumps(request.context, ensure_ascii=False, indent=2)}"

        child_user = (
            f"子任务：\n{request.task.strip()}\n\n"
            f"期望输出：{request.expected_output or '无特殊格式要求'}"
            f"{context_str}"
        )
        child_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _CHILD_SYSTEM_PROMPT},
            {"role": "user", "content": child_user},
        ]

        child_deadline = time.monotonic() + child_seconds
        logger.info(
            "spawn_agent 启动子任务 task_id=%s budget=%.1fs task=%.60s...",
            task_id,
            child_seconds,
            request.task,
        )

        # ── 调用子 ReAct 循环 ───────────────────────────────────────────────────
        answer, confidence = await run_loop_fn(
            messages=child_messages,
            serena_client=serena_client,
            acknowledge_cb=None,
            depth=1,
            deadline=child_deadline,
        )

        elapsed = time.monotonic() - started

        # 将 fallback 终止标记转换为结构化错误
        if isinstance(answer, str) and answer.startswith("__REACT_"):
            error_code = (
                SpawnErrorCode.SUBAGENT_TIMEOUT
                if answer in {"__REACT_TIMEOUT_BUDGET__", "__REACT_TIMEOUT__"}
                else SpawnErrorCode.SUBAGENT_INTERNAL_ERROR
            )
            return _make_failed(
                task_id=task_id,
                elapsed=elapsed,
                code=error_code,
                message=f"子 Agent 未正常 deliver，终止原因: {answer}",
            )

        logger.info(
            "spawn_agent 子任务完成 task_id=%s elapsed=%.1fs confidence=%s",
            task_id,
            elapsed,
            confidence,
        )
        return SpawnResult(
            ok=True,
            task_id=task_id,
            status=SpawnStatus.COMPLETED,
            answer=str(answer),
            confidence=str(confidence) if confidence else "medium",
            elapsed_seconds=elapsed,
            error_code=None,
            error_message=None,
        )

    except Exception as exc:
        return _make_failed(
            task_id=task_id,
            elapsed=time.monotonic() - started,
            code=SpawnErrorCode.SUBAGENT_INTERNAL_ERROR,
            message=f"invoke_spawn 内部异常: {exc}",
        )
