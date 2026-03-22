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
from pce.models import InsightConfidence, InsightFact, ModuleDigestDelta


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def _test_prompt_guidance() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="module:core-module",
                kind="module",
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
    _assert("read_digest_delta" in prompt, "system prompt 未要求先读取完整 digest_delta")
    _assert("mark_task_skipped" in prompt, "system prompt 未提示预算不足时可跳过任务")
    _assert(
        "changed_files_count=0 且 related_insights_count>0" in prompt,
        "system prompt 未强调 insight-only 任务应优先直接内化",
    )


async def _test_atomic_write_and_mark_done() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="module:core-module",
                kind="module",
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
                        "task_id": "module:core-module",
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


async def _test_task_summary_and_digest_delta_split() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="module:core-module",
                kind="module",
                status="pending",
                module_slug="core-module",
                digest_delta=ModuleDigestDelta(
                    module_id="11111111-1111-1111-1111-111111111111",
                    module_slug="core-module",
                    module_name="Core Module",
                    annotation_baseline="# Core Module\n",
                    related_insights=[
                        InsightFact(
                            id="22222222-2222-2222-2222-222222222222",
                            scope="pce/core.py",
                            content="需要修正核心职责描述",
                            confidence=InsightConfidence.HIGH,
                            created_at=datetime.now(UTC),
                        )
                    ],
                    changed_files=[],
                    external_context=[],
                ),
            )
        ],
        warnings=[],
        created_at=datetime.now(UTC),
    )

    summary = task_list.to_summary_dict()
    _assert("digest_delta" not in summary["items"][0], "任务摘要不应直接携带完整 digest_delta")
    _assert(summary["items"][0]["related_insights_count"] == 1, "任务摘要应保留 insight 数量")
    _assert(summary["items"][0]["changed_files_count"] == 0, "任务摘要应正确暴露 changed_files_count")
    _assert(summary["items"][0]["insight_only"] is True, "insight-only 任务应被正确标记")
    _assert(
        summary["items"][0]["insight_scopes_preview"] == ["pce/core.py"],
        "任务摘要应暴露 insight scope 预览",
    )
    _assert("temporal_stale" not in summary["items"][0], "任务摘要不应再暴露旧的 temporal_stale 字段")
    _assert("dirty_files" not in summary["items"][0], "任务摘要不应再暴露旧的 dirty_files 字段")

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
            "id": "tc-3",
            "function": {
                "name": "read_digest_delta",
                "arguments": json.dumps({"task_id": "module:core-module"}, ensure_ascii=False),
            },
        }
        tool_msg, summary_out = await agent._handle_virtual_tool(tool_call)
        _assert(summary_out is None, "read_digest_delta 不应直接 deliver")
        _assert(tool_msg is not None, "read_digest_delta 应返回 tool result")
        payload = json.loads(tool_msg["content"])
        _assert(payload["module_slug"] == "core-module", "read_digest_delta 应返回完整事实包")


async def _test_extension_heading_guard() -> None:
    task_list = DigestTaskList(
        items=[
            DigestTaskItem(
                id="module:core-module",
                kind="module",
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
            "id": "tc-2",
            "function": {
                "name": "write_annotation_patch",
                "arguments": json.dumps(
                    {
                        "module_slug": "core-module",
                        "operation": "append",
                        "target": "随意扩展标题",
                        "content": "- 不应允许的自由扩展标题。",
                    },
                    ensure_ascii=False,
                ),
            },
        }
        tool_msg, summary = await agent._handle_virtual_tool(tool_call)
        _assert(summary is None, "非法扩展标题不应直接 deliver")
        _assert(tool_msg is not None, "非法扩展标题应返回 tool result")
        _assert("扩展章节标题必须使用受控前缀" in tool_msg["content"], "未拦截非法扩展标题")


async def main() -> None:
    await _test_prompt_guidance()
    await _test_atomic_write_and_mark_done()
    await _test_task_summary_and_digest_delta_split()
    await _test_extension_heading_guard()
    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "system prompt 包含原子化收尾、digest_delta 读取和 insight-only 内化指引",
                    "write_annotation_and_mark_done 可同时写 annotation 与持久化 done 状态",
                    "read_task_list 与 read_digest_delta 已完成摘要/明细分离，并移除旧字段摘要",
                    "annotation 扩展标题受控前缀约束已生效",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
