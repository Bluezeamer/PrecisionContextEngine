"""SubAgent 运行时协议：数据模型、错误码与工具 schema。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 状态与错误码 ───────────────────────────────────────────────────────────────

class SpawnStatus(str, Enum):
    """SubAgent 执行终态。"""

    COMPLETED = "completed"
    FAILED = "failed"


class SpawnErrorCode(str, Enum):
    """SubAgent 失败原因码，便于父 Agent 决策是否重试或降级。"""

    SUBAGENT_TIMEOUT = "SUBAGENT_TIMEOUT"
    """子 Agent 超出时间预算。"""
    SUBAGENT_DEPTH_EXCEEDED = "SUBAGENT_DEPTH_EXCEEDED"
    """调用方已是子 Agent，禁止递归 spawn。"""
    SUBAGENT_BUDGET_REJECTED = "SUBAGENT_BUDGET_REJECTED"
    """父剩余预算不足以满足最小子预算。"""
    SUBAGENT_INVALID_ARGS = "SUBAGENT_INVALID_ARGS"
    """spawn_agent 调用参数格式错误。"""
    SUBAGENT_INTERNAL_ERROR = "SUBAGENT_INTERNAL_ERROR"
    """子 Agent 内部异常（非超时）。"""


# ── 请求 / 结果数据模型 ────────────────────────────────────────────────────────

@dataclass(slots=True)
class SpawnRequest:
    """父 Agent 发起 spawn 调用时的参数，对应 spawn_agent 工具 arguments。"""

    task: str
    """子任务描述，必须可独立执行。"""
    allocated_seconds: float
    """希望分配给子 Agent 的时间预算（秒）。实际值由 spawner 裁剪。"""
    expected_output: str = ""
    """期望输出格式或结构描述（可选），注入子 Agent system prompt。"""
    context: dict[str, Any] = field(default_factory=dict)
    """可序列化的补充上下文（可选），注入子 Agent user 消息。"""
    strict: bool = False
    """严格模式：子失败时父 Agent 也应停止（由父 Agent 自行判断，spawner 不强制）。"""


@dataclass(slots=True)
class SpawnResult:
    """spawn_agent 工具调用的执行结果，作为 tool result 回注父消息。"""

    ok: bool
    task_id: str
    status: SpawnStatus
    answer: str
    confidence: str
    elapsed_seconds: float
    error_code: SpawnErrorCode | None
    error_message: str | None

    def to_tool_content(self) -> dict[str, Any]:
        """序列化为 tool result content（JSON-safe dict），回注父消息历史。"""
        return {
            "ok": self.ok,
            "task_id": self.task_id,
            "status": self.status.value,
            "answer": self.answer,
            "confidence": self.confidence,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "error_code": self.error_code.value if self.error_code is not None else None,
            "error_message": self.error_message,
        }


# ── 预算常量 ───────────────────────────────────────────────────────────────────

MIN_SPAWN_BUDGET: float = 8.0
"""子 Agent 最小时间预算（秒）。父剩余不足此值时拒绝 spawn。"""

MAX_SPAWN_BUDGET_RATIO: float = 0.70
"""子 Agent 最多可占父剩余预算的比例。超出部分自动向下裁剪。"""

MAX_SPAWNS_PER_LOOP: int = 3
"""单次 ReAct 循环内允许的最大 spawn 次数（子 Agent 独立会话不计入）。"""


# ── spawn_agent 工具 schema ────────────────────────────────────────────────────

SPAWN_AGENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "spawn_agent",
        "description": (
            "将一个独立子问题委托给子 Agent 推理。"
            "子 Agent 不能再 spawn，完成后返回结构化 JSON 结果。"
            "使用场景：当前子任务可独立检索、与主线无强依赖时再使用；"
            "对于简单查询，直接调用 Serena 工具效率更高。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "子任务描述，必须是自包含的、可独立执行的指令。",
                },
                "allocated_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "description": "期望分配给子 Agent 的时间预算（秒）。实际值可能被父预算裁剪。",
                },
                "expected_output": {
                    "type": "string",
                    "description": "期望输出格式或结构说明（可选）。",
                    "default": "",
                },
                "context": {
                    "type": "object",
                    "description": "传递给子 Agent 的可序列化补充上下文（可选）。",
                    "default": {},
                },
                "strict": {
                    "type": "boolean",
                    "description": "是否要求子任务必须成功（可选，默认 false）。",
                    "default": False,
                },
            },
            "required": ["task", "allocated_seconds"],
        },
    },
}
