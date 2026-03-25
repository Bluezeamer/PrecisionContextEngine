"""
Prompt guard 轻量回归脚本。

目标：
1. 验证 query 初始输入超限时会触发降级；
2. 验证 digest system prompt 可降级到 compact/minimal；
3. 验证 annotation_writer 会主动压缩 prompt 负载。

运行：
    uv run python scripts/test_prompt_guard.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pce.agent as agent_module
import pce.annotation_writer as annotation_writer_module
from pce.agent import PCEAgent, _SYSTEM_PROMPT_PLACEHOLDER
from pce.annotation_writer import (
    _build_index_md_prompt,
    _build_missing_coverage_repair_prompt,
    _llm_complete_text,
)
from pce.digest_agent import DigestAgent, DigestTaskItem, DigestTaskList
from pce.models import FileMeta, IndexEntry
from pce.prompt_guard import build_prompt_budget, estimate_input_tokens


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _entry(path: str, loc: int) -> IndexEntry:
    return IndexEntry(
        file_meta=FileMeta(
            path=Path(path),
            language="python",
            size_bytes=256,
            mtime=datetime.now(UTC),
            loc=loc,
        ),
        symbols=[],
        imports=[],
        edges=[],
    )


async def _test_query_guard() -> None:
    agent = PCEAgent(model="gpt-4o-mini", provider="openai")
    system = "S" * 220000
    question = "Q" * 120000
    original_window = agent_module._CONTEXT_WINDOW
    agent_module._CONTEXT_WINDOW = 8000
    try:
        guarded_system, guarded_user = agent._guard_query_prompt(
            system_content=system,
            question=question,
            tools_schema=[],
        )
    finally:
        agent_module._CONTEXT_WINDOW = original_window
    _assert(guarded_user != question, "query guard 未压缩超长用户输入")
    _assert(
        guarded_system == _SYSTEM_PROMPT_PLACEHOLDER or _SYSTEM_PROMPT_PLACEHOLDER in guarded_system,
        "query guard 在极端超限时应回退到 placeholder system prompt",
    )


async def _test_digest_guard() -> None:
    items = [
        DigestTaskItem(
            id=f"module:mod-{idx}",
            kind="module",
            status="pending",
            module_slug=f"mod-{idx}",
        )
        for idx in range(40)
    ]
    task_list = DigestTaskList(
        items=items,
        warnings=["warn"] * 20,
        created_at=datetime.now(UTC),
    )

    class _FakeSerena:
        tools_schema: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmp:
        agent = DigestAgent(
            project_root=Path(tmp),
            task_list_path=Path(tmp) / ".pce" / "digest_tasks.json",
            model="gpt-4o-mini",
            provider="openai",
        )
        full_prompt = agent._build_system_prompt(task_list, detail="full")
        guarded = agent._guard_system_prompt(task_list, _FakeSerena(), full_prompt)
        _assert(guarded, "digest guard 不应返回空 prompt")
        _assert("## 当前任务" in guarded or "## 当前任务概览" in guarded, "digest guard 输出缺少任务区块")


async def _test_annotation_prompt_compaction() -> None:
    entries = [_entry(f"pkg/file_{idx}.py", 50 + idx) for idx in range(120)]
    prompt = _build_index_md_prompt(
        entries,
        project_meta=type("ProjectMetaLike", (), {
            "root_path": Path("/tmp/demo"),
            "file_count": len(entries),
            "loc_total": sum(item.file_meta.loc for item in entries),
        })(),
    )
    budget = build_prompt_budget()
    tokens = estimate_input_tokens(
        "gpt-4o-mini",
        [
            {"role": "system", "content": "annotation-writer-budget-check"},
            {"role": "user", "content": prompt},
        ],
    )
    _assert(tokens <= budget.target_input_budget, "index prompt 未主动收敛到目标预算内")

    facts = [
        {
            "path": f"pkg/file_{idx}.py",
            "language": "python",
            "loc": 200,
            "symbol_summary": [f"symbol_{j}" for j in range(8)],
            "content_windows": {"full": "line\n" * 80},
        }
        for idx in range(8)
    ]
    repair_prompt = _build_missing_coverage_repair_prompt(
        facts,
        index_content=prompt,
        known_slugs={f"slug-{idx}" for idx in range(20)},
    )
    payload = {
        "repair_prompt_tokens": estimate_input_tokens(
            "gpt-4o-mini",
            [
                {"role": "system", "content": "annotation-writer-budget-check"},
                {"role": "user", "content": repair_prompt},
            ],
        )
    }
    _assert(payload["repair_prompt_tokens"] <= build_prompt_budget().hard_input_budget, "遗漏补归属 prompt 仍然明显超限")


async def _test_annotation_failure_logging() -> None:
    messages: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    def _boom(**_: object) -> object:
        raise TimeoutError()

    handler = _ListHandler()
    logger = logging.getLogger(annotation_writer_module.__name__)
    logger.addHandler(handler)
    original = annotation_writer_module.litellm.completion
    annotation_writer_module.litellm.completion = _boom
    try:
        result = await _llm_complete_text(
            "prompt",
            system_prompt="sys",
            model="gpt-4o-mini",
            failure_log="annotation smoke",
        )
    finally:
        annotation_writer_module.litellm.completion = original
        logger.removeHandler(handler)

    _assert(result is None, "失败时应降级返回 None")
    _assert(
        any("TimeoutError" in message for message in messages),
        "降级日志至少应包含异常类型",
    )


async def main() -> None:
    await _test_query_guard()
    await _test_digest_guard()
    await _test_annotation_prompt_compaction()
    await _test_annotation_failure_logging()
    print(json.dumps({"ok": True, "tests": ["query guard", "digest guard", "annotation prompt compaction", "annotation failure logging"]}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
