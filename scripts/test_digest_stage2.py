"""Digest stage2 最小回归测试。

运行：
    uv run python scripts/test_digest_stage2.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from pce.digest_cognition_agent import (
    DigestFilterStageAgent,
    DigestStaleCheckStageAgent,
    SharedToolBudget,
    build_stale_check_facts_text,
)
from pce.memory import save_file_baseline
from pce.models import InsightConfidence, InsightFact


def _insight(idx: int, *, scope: str, content: str) -> InsightFact:
    return InsightFact(
        id=f"00000000-0000-0000-0000-{idx:012d}",
        scope=scope,
        content=content,
        confidence=InsightConfidence.MEDIUM,
        created_at=datetime.now(UTC),
    )


async def _call_virtual(agent, name: str, args: dict) -> str:
    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": name, "arguments": __import__("json").dumps(args, ensure_ascii=False)},
    }
    result = await agent.handle_virtual_tool(tool_call, state=None)  # type: ignore[arg-type]
    assert result is not None
    return str(result.get("content") or "")


async def _test_shared_budget_between_stage1_and_stage2() -> None:
    with TemporaryDirectory(prefix="pce-digest-stage2-budget-") as tmpdir:
        root = Path(tmpdir)
        (root / ".pce" / "annotations" / "modules").mkdir(parents=True, exist_ok=True)
        (root / ".pce" / "annotations" / "modules" / "alpha.md").write_text(
            "# Alpha\n\n## 覆盖文件\n- src/alpha.py\n",
            "utf-8",
        )
        await save_file_baseline(
            "src/alpha.py",
            content="def run():\n    return 1\n",
            content_hash="baseline-hash",
            symbols=[],
            root_path=root,
        )
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "alpha.py").write_text("def run():\n    return 2\n", "utf-8")

        insights = [_insight(1, scope="src/alpha.py", content="alpha 模块负责执行 run 流程")]
        shared = SharedToolBudget(total=3)

        stage1 = DigestFilterStageAgent(
            project_root=root,
            insights=insights,
            facts_text="stage1 facts",
            facts_truncated=False,
            shared_budget=shared,
        )
        content = await _call_virtual(stage1, "read_annotation", {"path": ".pce/annotations/modules/alpha.md"})
        assert "覆盖文件" in content
        assert shared.used == 1

        facts_text, truncated = build_stale_check_facts_text(
            insights=insights,
            dirty_files=["src/alpha.py"],
            model="gpt-4o-mini",
        )
        stage2 = DigestStaleCheckStageAgent(
            project_root=root,
            insights=insights,
            facts_text=facts_text,
            facts_truncated=truncated,
            dirty_files=["src/alpha.py"],
            shared_budget=shared,
        )
        diff_content = await _call_virtual(stage2, "read_file_diff", {"paths": ["src/alpha.py"]})
        assert "a/src/alpha.py" in diff_content
        assert "b/src/alpha.py" in diff_content
        assert shared.used == 2


async def _test_stage2_rejects_non_dirty_paths() -> None:
    with TemporaryDirectory(prefix="pce-digest-stage2-guard-") as tmpdir:
        root = Path(tmpdir)
        insights = [_insight(2, scope="src/alpha.py", content="alpha insight")]
        facts_text, truncated = build_stale_check_facts_text(
            insights=insights,
            dirty_files=["src/alpha.py"],
            model="gpt-4o-mini",
        )
        agent = DigestStaleCheckStageAgent(
            project_root=root,
            insights=insights,
            facts_text=facts_text,
            facts_truncated=truncated,
            dirty_files=["src/alpha.py"],
            shared_budget=SharedToolBudget(total=3),
        )
        content = await _call_virtual(agent, "read_file_diff", {"paths": ["src/other.py"]})
        assert "没有可读取的 dirty file diff" in content


def main() -> None:
    asyncio.run(_test_shared_budget_between_stage1_and_stage2())
    asyncio.run(_test_stage2_rejects_non_dirty_paths())
    print("digest stage2 tests passed")


if __name__ == "__main__":
    main()
