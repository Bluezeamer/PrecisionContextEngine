"""
DigestAgent 轻量回归脚本。

目标：
1. 验证 system prompt 已明确引导优先使用原子化收尾工具；
2. 验证 write_annotation_and_mark_done 会同时写 annotation 与持久化任务状态。

运行：
    uv run python scripts/test_digest_agent.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pce.digest_agent import DigestAgent, DigestTaskItem, DigestTaskList


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _test_prompt_guidance() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="insight:1",
                kind="insight",
                status="pending",
                module_slug="core-module",
            )
        ],
        warnings=[],
        created_at=datetime.now(UTC),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        agent = DigestAgent(
            project_root=root,
            task_list_path=root / ".pce" / "digest_tasks.json",
            model="dummy-model",
            provider="openai",
        )
        prompt = agent._build_system_prompt(task_list)

    _assert(
        "write_annotation_and_mark_done" in prompt,
        "system prompt 未强调原子化收尾工具",
    )
    _assert("mark_task_skipped" in prompt, "system prompt 未提示预算不足时可跳过任务")


async def _test_atomic_write_and_mark_done() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="insight:task-1",
                kind="insight",
                status="pending",
                module_slug="core-module",
            )
        ],
        warnings=[],
        created_at=datetime.now(UTC),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task_list_path = root / ".pce" / "digest_tasks.json"
        agent = DigestAgent(
            project_root=root,
            task_list_path=task_list_path,
            model="dummy-model",
            provider="openai",
        )
        agent._task_list = task_list

        tool_call = {
            "id": "tc-1",
            "function": {
                "name": "write_annotation_and_mark_done",
                "arguments": json.dumps(
                    {
                        "task_id": "insight:task-1",
                        "note": "已整合到核心职责章节",
                        "operation": "append",
                        "target": "核心职责",
                        "content": "- DigestAgent 优先使用原子化收尾工具，避免遗漏 mark done。",
                    },
                    ensure_ascii=False,
                ),
            },
        }

        tool_msg, summary = await agent._handle_virtual_tool(tool_call)
        _assert(summary is None, "原子化收尾工具不应直接 deliver")
        _assert(tool_msg is not None, "原子化收尾工具应返回 tool result")
        _assert("完成任务" in tool_msg["content"], "tool result 未体现任务已完成")

        annotation_path = root / ".pce" / "annotations" / "modules" / "core-module.md"
        annotation = annotation_path.read_text(encoding="utf-8")
        _assert("## 核心职责" in annotation, "annotation 缺少固定章节")
        _assert("原子化收尾工具" in annotation, "annotation 未写入预期内容")

        payload = json.loads(task_list_path.read_text(encoding="utf-8"))
        item = payload["items"][0]
        _assert(item["status"] == "done", "任务状态未持久化为 done")
        _assert(item["note"] == "已整合到核心职责章节", "任务 note 未持久化")


async def main() -> None:
    await _test_prompt_guidance()
    await _test_atomic_write_and_mark_done()
    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "system prompt 包含原子化收尾与预算收敛指引",
                    "write_annotation_and_mark_done 可同时写 annotation 与持久化 done 状态",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
