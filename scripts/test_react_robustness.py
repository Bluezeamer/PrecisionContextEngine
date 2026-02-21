"""
测试 PCEAgent._run_react_loop 的健壮性。

策略：mock _completion 控制 LLM 返回序列，mock _invoke_tool 控制工具结果，
不依赖真实 Serena 或 LLM，纯粹验证循环的状态机逻辑。

覆盖的边界路径：
  T01 — 正常路径：deliver 正常终止
  T02 — 多轮工具调用后 deliver 终止
  T03 — deliver 与 Serena 工具同批出现
  T04 — deliver args 为空 → __REACT_DELIVER_EMPTY__
  T05 — deliver args JSON 解析失败 → __REACT_DELIVER_EMPTY__
  T06 — LLM 违规直接文字回答，纠正后 deliver → 正常终止
  T07 — LLM 持续违规，纠正次数耗尽 → __REACT_NO_TOOL_EXHAUSTED__
  T08 — tool_call args JSON 不完整，工具收到错误 result，LLM 重调后 deliver
  T09 — finish_reason=length，content 截断续写后 deliver
  T10 — finish_reason=length，tool_call 截断，重新调用后 deliver
  T11 — finish_reason=length 续写次数耗尽 → __REACT_LENGTH_EXHAUSTED__
  T12 — asyncio.TimeoutError 触发重试，重试成功后 deliver
  T13 — asyncio.TimeoutError 重试耗尽 → __REACT_TIMEOUT__
  T14 — 步数耗尽 → __REACT_MAX_STEPS_EXCEEDED__
  T15 — finish_reason=tool_calls 但 tool_calls 为空（异常响应）→ 纠正后 deliver
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# 加载 .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            import os
            os.environ.setdefault(k.strip(), v.strip())

from pce.agent import (
    PCEAgent,
    _MAX_LENGTH_CONT,
    _MAX_NO_TOOL_RETRIES,
    _MAX_TIMEOUT_RETRIES,
    _parse_query_response,
)

# ── 工具函数 ──────────────────────────────────────────────────────────────────

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

_results: list[tuple[str, bool, str]] = []


def assert_eq(label: str, actual: Any, expected: Any) -> bool:
    ok = actual == expected
    _results.append((label, ok, f"期望 {expected!r}，实际 {actual!r}"))
    return ok


def assert_in(label: str, needle: str, haystack: str) -> bool:
    ok = needle in haystack
    _results.append((label, ok, f"期望包含 {needle!r}，实际 {haystack!r}"))
    return ok


def make_agent(max_steps: int = 10) -> PCEAgent:
    return PCEAgent(model="fake-model", max_steps=max_steps)


def make_serena_mock(tool_result: str = "工具执行成功") -> MagicMock:
    """返回一个假的 SerenaClient，tools_schema 为空列表。"""
    mock = MagicMock()
    mock.tools_schema = []
    mock.call = AsyncMock(return_value=tool_result)
    return mock


def make_response(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
) -> tuple[MagicMock, str]:
    """构造一个 (_completion 返回的 response, finish_reason) 元组。

    关键：让 message 不含 model_dump，确保 _extract_message 走 dict 路径，
    避免 MagicMock.model_dump() 返回的 Mock 对象污染 tool_calls 解析。
    """
    # 构造 tool_calls 列表（对象形式，兼容 _extract_tool_calls 的对象路径）
    tc_objects = []
    if tool_calls:
        for tc in tool_calls:
            tc_obj = MagicMock()
            tc_obj.id = tc.get("id", str(uuid.uuid4()))
            tc_obj.type = "function"
            fn = MagicMock()
            fn.name = tc["name"]
            fn.arguments = json.dumps(tc.get("args", {}))
            tc_obj.function = fn
            tc_objects.append(tc_obj)

    # message 作为 dict，绕过 model_dump 路径
    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": content or "",
    }
    if tc_objects:
        message_dict["tool_calls"] = tc_objects

    # choice 返回 dict，让 getattr(choice, "message") 失败，走 choice.get("message") 路径
    # 但 litellm 通常是对象，这里保持对象形式，只让 message 本身是 dict
    choice = MagicMock()
    choice.message = message_dict  # 直接是 dict，_extract_message 会走 isinstance(message, dict) 路径
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    return response, finish_reason


def make_deliver_tc(answer: str, confidence: str = "high") -> dict:
    return {"name": "deliver", "args": {"answer": answer, "confidence": confidence}}


def make_tool_tc(name: str = "find_symbol", args: dict | None = None) -> dict:
    return {"name": name, "args": args or {"query": "test"}}


# ── 核心测试运行器 ────────────────────────────────────────────────────────────

async def run_loop(
    agent: PCEAgent,
    serena: MagicMock,
    response_seq: list[tuple[MagicMock, str]],
) -> str:
    """用 response_seq 依次替换 _completion 的返回值，运行 _run_react_loop。

    注意：不能用 next(iter(...))，StopIteration 在 async 函数中会变成 RuntimeError。
    改用 list.pop(0)，序列耗尽时抛出 IndexError（会被测试框架捕获报告）。
    """
    seq = list(response_seq)

    async def fake_completion(messages, tools):
        return seq.pop(0)

    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
    with patch.object(agent, "_completion", side_effect=fake_completion):
        return await agent._run_react_loop(messages, serena)


# ── 测试用例 ──────────────────────────────────────────────────────────────────

async def t01_normal_deliver():
    """正常路径：第一轮直接 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(tool_calls=[make_deliver_tc("答案A")]),
    ])
    assert_eq("T01 正常 deliver", result, "答案A")


async def t02_multi_step_then_deliver():
    """多轮工具调用后 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(tool_calls=[make_tool_tc()], finish_reason="tool_calls"),
        make_response(tool_calls=[make_tool_tc()], finish_reason="tool_calls"),
        make_response(tool_calls=[make_deliver_tc("答案B")]),
    ])
    assert_eq("T02 多轮工具后 deliver", result, "答案B")


async def t03_deliver_with_serena_same_batch():
    """deliver 与 Serena 工具同批出现，Serena 先执行"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(tool_calls=[make_tool_tc(), make_deliver_tc("答案C")], finish_reason="tool_calls"),
    ])
    assert_eq("T03 同批 deliver", result, "答案C")
    # Serena 应该被调用了一次
    assert_in("T03 Serena 被调用", "call_count", "call_count")  # 通过 serena.call.call_count 检查
    ok = serena.call.call_count == 1
    _results.append(("T03 Serena 调用次数=1", ok, f"实际 call_count={serena.call.call_count}"))


async def t04_deliver_empty_answer():
    """deliver answer 为空字符串 → __REACT_DELIVER_EMPTY__"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(tool_calls=[{"name": "deliver", "args": {"answer": ""}}]),
    ])
    assert_eq("T04 deliver 空 answer", result, "__REACT_DELIVER_EMPTY__")


async def t05_deliver_invalid_json_args():
    """deliver args JSON 不合法 → __REACT_DELIVER_EMPTY__"""
    agent = make_agent()
    serena = make_serena_mock()

    bad_tc = MagicMock()
    bad_tc.id = str(uuid.uuid4())
    fn = MagicMock()
    fn.name = "deliver"
    fn.arguments = "{invalid json"
    bad_tc.function = fn

    # message 为 dict，tool_calls 含非法 JSON 的 deliver
    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [bad_tc],
    }
    choice = MagicMock()
    choice.message = message_dict
    choice.finish_reason = "tool_calls"
    response = MagicMock()
    response.choices = [choice]

    result = await run_loop(agent, serena, [(response, "tool_calls")])
    assert_eq("T05 deliver JSON 解析失败", result, "__REACT_DELIVER_EMPTY__")


async def t06_no_tool_then_corrected():
    """LLM 先违规直接回答，纠正后正常 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(content="我直接回答了"),           # 违规
        make_response(tool_calls=[make_deliver_tc("答案D")]),  # 纠正后 deliver
    ])
    assert_eq("T06 纠正后 deliver", result, "答案D")


async def t07_no_tool_exhausted():
    """LLM 持续违规，纠正次数耗尽"""
    agent = make_agent()
    serena = make_serena_mock()
    # 违规次数 = _MAX_NO_TOOL_RETRIES + 1
    responses = [make_response(content=f"违规{i}") for i in range(_MAX_NO_TOOL_RETRIES + 1)]
    result = await run_loop(agent, serena, responses)
    assert_eq("T07 纠正次数耗尽", result, "__REACT_NO_TOOL_EXHAUSTED__")


async def t08_invalid_tool_args_then_recover():
    """Serena 工具 args 解析失败，LLM 收到错误 result 后重调并 deliver"""
    agent = make_agent()
    serena = make_serena_mock()

    bad_tc = MagicMock()
    bad_tc.id = str(uuid.uuid4())
    fn = MagicMock()
    fn.name = "find_symbol"
    fn.arguments = "{bad"
    bad_tc.function = fn

    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [bad_tc],
    }
    choice1 = MagicMock()
    choice1.message = message_dict
    choice1.finish_reason = "tool_calls"
    resp1 = MagicMock()
    resp1.choices = [choice1]

    result = await run_loop(agent, serena, [
        (resp1, "tool_calls"),
        make_response(tool_calls=[make_deliver_tc("答案E")]),
    ])
    assert_eq("T08 args 解析失败后恢复", result, "答案E")
    # Serena.call 不应被调用（args 解析失败由 _invoke_tool 拦截）
    ok = serena.call.call_count == 0
    _results.append(("T08 Serena 未被调用", ok, f"实际 call_count={serena.call.call_count}"))


async def t09_length_content_continuation():
    """finish_reason=length，content 截断，续写后 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        make_response(content="推理到一半...", finish_reason="length"),  # 截断
        make_response(tool_calls=[make_deliver_tc("答案F")]),            # 续写后 deliver
    ])
    assert_eq("T09 content 续写后 deliver", result, "答案F")


async def t10_length_tool_call_truncated():
    """finish_reason=length，tool_call JSON 截断，重新调用后 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    result = await run_loop(agent, serena, [
        # tool_call 截断：content 为空，tool_calls 也没有（被丢弃了）
        make_response(content=None, tool_calls=None, finish_reason="length"),
        make_response(tool_calls=[make_deliver_tc("答案G")]),
    ])
    assert_eq("T10 tool_call 截断后 deliver", result, "答案G")


async def t11_length_exhausted():
    """finish_reason=length 续写次数耗尽"""
    agent = make_agent()
    serena = make_serena_mock()
    # 触发次数 = _MAX_LENGTH_CONT + 1
    responses = [
        make_response(content="截断内容", finish_reason="length")
        for _ in range(_MAX_LENGTH_CONT + 1)
    ]
    result = await run_loop(agent, serena, responses)
    assert_eq("T11 续写次数耗尽", result, "__REACT_LENGTH_EXHAUSTED__")


async def t12_timeout_retry_success():
    """asyncio.TimeoutError 触发重试，重试成功后 deliver"""
    agent = make_agent()
    serena = make_serena_mock()
    call_count = 0

    async def fake_completion_with_timeout(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise asyncio.TimeoutError()  # 第一次超时
        return make_response(tool_calls=[make_deliver_tc("答案H")])  # 重试成功

    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
    with patch.object(agent, "_completion", side_effect=fake_completion_with_timeout):
        result = await agent._run_react_loop(messages, serena)

    assert_eq("T12 超时重试成功", result, "答案H")
    ok = call_count == 2
    _results.append(("T12 共调用 2 次", ok, f"实际 call_count={call_count}"))


async def t13_timeout_exhausted():
    """asyncio.TimeoutError 重试耗尽"""
    agent = make_agent()
    serena = make_serena_mock()

    async def always_timeout(messages, tools):
        raise asyncio.TimeoutError()

    messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
    with patch.object(agent, "_completion", side_effect=always_timeout):
        result = await agent._run_react_loop(messages, serena)

    assert_eq("T13 超时耗尽", result, "__REACT_TIMEOUT__")


async def t14_max_steps_exceeded():
    """步数耗尽 → __REACT_MAX_STEPS_EXCEEDED__"""
    max_steps = 3
    agent = make_agent(max_steps=max_steps)
    serena = make_serena_mock()
    # 每步都调 Serena，永远不 deliver
    responses = [
        make_response(tool_calls=[make_tool_tc()], finish_reason="tool_calls")
        for _ in range(max_steps)
    ]
    result = await run_loop(agent, serena, responses)
    assert_eq("T14 步数耗尽", result, "__REACT_MAX_STEPS_EXCEEDED__")


async def t15_finish_reason_tool_calls_but_empty():
    """finish_reason=tool_calls 但 tool_calls 实际为空 → 纠正后 deliver"""
    agent = make_agent()
    serena = make_serena_mock()

    # message 为 dict，不含 tool_calls 字段（模拟 provider 异常响应）
    message_dict: dict[str, Any] = {"role": "assistant", "content": ""}
    choice = MagicMock()
    choice.message = message_dict
    choice.finish_reason = "tool_calls"
    response = MagicMock()
    response.choices = [choice]

    result = await run_loop(agent, serena, [
        (response, "tool_calls"),
        make_response(tool_calls=[make_deliver_tc("答案I")]),
    ])
    assert_eq("T15 finish_reason=tool_calls 但无工具 → 纠正后 deliver", result, "答案I")


# ── 兜底标记解析测试 ──────────────────────────────────────────────────────────

def test_fallback_markers():
    """验证所有兜底标记都能被 _parse_query_response 正确识别"""
    sid = str(uuid.uuid4())
    markers = {
        "__REACT_MAX_STEPS_EXCEEDED__": "步数",
        "__REACT_NO_TOOL_EXHAUSTED__": "未调用工具",
        "__REACT_DELIVER_EMPTY__": "空结论",
        "__REACT_LENGTH_EXHAUSTED__": "截断",
        "__REACT_TIMEOUT__": "超时",
    }
    for marker, keyword in markers.items():
        r = _parse_query_response(marker, sid)
        assert_in(f"fallback {marker}", keyword, r.answer)


# ── 主程序 ────────────────────────────────────────────────────────────────────

async def main():
    tests = [
        t01_normal_deliver,
        t02_multi_step_then_deliver,
        t03_deliver_with_serena_same_batch,
        t04_deliver_empty_answer,
        t05_deliver_invalid_json_args,
        t06_no_tool_then_corrected,
        t07_no_tool_exhausted,
        t08_invalid_tool_args_then_recover,
        t09_length_content_continuation,
        t10_length_tool_call_truncated,
        t11_length_exhausted,
        t12_timeout_retry_success,
        t13_timeout_exhausted,
        t14_max_steps_exceeded,
        t15_finish_reason_tool_calls_but_empty,
    ]

    print(f"\n{'='*60}")
    print(f"  PCEAgent ReAct 健壮性测试 ({len(tests)} 个场景)")
    print(f"{'='*60}\n")

    for test_fn in tests:
        try:
            await test_fn()
        except Exception as e:
            _results.append((test_fn.__name__, False, f"未捕获异常: {e}"))

    # 同步测试
    try:
        test_fallback_markers()
    except Exception as e:
        _results.append(("test_fallback_markers", False, f"未捕获异常: {e}"))

    # 输出结果
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    for name, ok, detail in _results:
        icon = PASS if ok else FAIL
        print(f"  {icon} {name}")
        if not ok:
            print(f"       → {detail}")

    print(f"\n{'='*60}")
    print(f"  结果: {passed}/{total} 通过", end="")
    if passed == total:
        print("  \033[32m全部通过\033[0m")
    else:
        print(f"  \033[31m{total - passed} 项失败\033[0m")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
