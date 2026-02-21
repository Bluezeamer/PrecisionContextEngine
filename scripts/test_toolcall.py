"""
验证 step-3.5-flash (via OpenRouter) 的 tool call 能力。
测试场景：
  1. 单次 tool call
  2. 多 tool call（并行）
  3. deliver 终止信号 toolcall（为 ReAct 改造预研）
"""

import json
import os
import sys
from pathlib import Path

# 加载 .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import litellm

MODEL = os.getenv("PCE_MODEL", "openrouter/stepfun/step-3.5-flash:free")

# ── 工具定义 ──────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取指定时区的当前时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "时区，如 Asia/Shanghai"},
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deliver",
            "description": "当你已完成所有推理和信息收集，调用此工具提交最终答案。必须在得出结论时调用，不能继续调用其他工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "最终答案内容"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "答案置信度",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ── 模拟工具执行 ──────────────────────────────────────────────
def mock_tool_call(name: str, args: dict) -> str:
    if name == "get_weather":
        return json.dumps({"city": args["city"], "temp": 22, "condition": "晴", "unit": args.get("unit", "celsius")})
    if name == "get_time":
        return json.dumps({"timezone": args["timezone"], "time": "14:30:00", "date": "2026-02-19"})
    return json.dumps({"error": f"未知工具: {name}"})


# ── 打印辅助 ──────────────────────────────────────────────────
def p(label: str, content):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if isinstance(content, (dict, list)):
        print(json.dumps(content, ensure_ascii=False, indent=2))
    else:
        print(content)


# ── 测试用例 ──────────────────────────────────────────────────
def run_test(title: str, messages: list, max_steps: int = 5):
    print(f"\n{'#'*60}")
    print(f"# 测试: {title}")
    print(f"{'#'*60}")

    for step in range(max_steps):
        print(f"\n[Step {step + 1}] 调用 LLM...")
        print(f"  messages 数量: {len(messages)}")
        for i, m in enumerate(messages):
            print(f"  [{i}] role={m.get('role')}, keys={list(m.keys())}")
        resp = litellm.completion(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
        )

        msg = resp.choices[0].message
        # litellm 返回对象，转为 dict 存入 messages
        msg_dict = {"role": "assistant"}
        content = getattr(msg, "content", None)
        # StepFun 要求 content 字段始终存在，即便是 null
        msg_dict["content"] = content or ""
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(msg_dict)

        if not tool_calls:
            p("LLM 直接回答（无 tool call）", content or "（空）")
            print("\n[结果] 循环以无 tool_call 方式结束")
            return

        p(f"LLM 调用了 {len(tool_calls)} 个工具", [
            {"name": tc.function.name, "args": tc.function.arguments} for tc in tool_calls
        ])

        # 检测 deliver 信号
        deliver_call = next(
            (tc for tc in tool_calls if tc.function.name == "deliver"), None
        )
        if deliver_call:
            args = json.loads(deliver_call.function.arguments)
            p("检测到 deliver 终止信号", args)
            print(f"\n[结果] 循环以 deliver 方式正常终止，答案: {args.get('answer')}")
            return

        # 执行普通工具并追加结果
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            result = mock_tool_call(tc.function.name, args)
            print(f"  → 执行 {tc.function.name}({args}) = {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                # StepFun 不识别 name 字段，仅保留 tool_call_id + content
                "content": result,
            })

    print(f"\n[结果] 达到最大步数 {max_steps}，强制终止（兜底）")


# ── 主程序 ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"模型: {MODEL}")
    print(f"litellm 已加载")
    # litellm._turn_on_debug()

    # 测试1：触发单次 tool call
    run_test(
        title="单次 tool call（查天气）",
        messages=[
            {"role": "user", "content": "北京现在天气怎么样？"},
        ],
    )

    # 测试2：触发多个并行 tool call
    run_test(
        title="多 tool call（天气 + 时间）",
        messages=[
            {"role": "user", "content": "告诉我北京的天气和上海时区的当前时间。"},
        ],
    )

    # 测试3：完整 ReAct 流程 + deliver 终止
    run_test(
        title="完整 ReAct + deliver 终止信号",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个助手，可以调用工具获取信息。"
                    "当你收集到足够信息并得出结论时，必须调用 deliver 工具提交最终答案，"
                    "不能只用文字回答，必须通过 deliver 交付。"
                ),
            },
            {"role": "user", "content": "查询北京天气和亚洲上海时区时间，然后给我一个综合总结。"},
        ],
    )
