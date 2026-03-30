"""Digest stage2 最小回归测试。

运行：
    uv run python scripts/test_digest_stage2.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pce.digest_agent as digest_agent
from pce.digest_cognition_agent import (
    DigestCleanupStageAgent,
    DigestFilterStageAgent,
    DigestStaleCheckStageAgent,
    SharedToolBudget,
    build_stale_check_facts_text,
)
from pce.insight_cache import InsightCache
from pce.memory import save_file_baseline, save_index
from pce.models import (
    BuildStats,
    FileMeta,
    IndexEntry,
    IndexSnapshot,
    InsightConfidence,
    InsightFact,
    NavigationTree,
    ProjectMeta,
)
from pce.staging import DirtyState


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


async def _test_stageC_rewrite_and_reset_tools() -> None:
    with TemporaryDirectory(prefix="pce-digest-stagec-") as tmpdir:
        root = Path(tmpdir)
        annotations = root / ".pce" / "annotations"
        modules_dir = annotations / "modules"
        areas_dir = annotations / "areas"
        modules_dir.mkdir(parents=True, exist_ok=True)
        areas_dir.mkdir(parents=True, exist_ok=True)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "alpha.py").write_text("def run():\n    return 2\n", "utf-8")

        module_path = modules_dir / "alpha.md"
        module_path.write_text(
            "# Alpha\n\n## 覆盖文件\n- src/alpha.py\n\n## 核心职责\n- 旧职责\n\n## 关键流程\n- 旧流程\n",
            "utf-8",
        )
        (areas_dir / "runtime.md").write_text("# 运行时\n\n## 区域说明\n旧区域说明\n", "utf-8")
        (annotations / "index.md").write_text("# 项目认知导航\n\n## 项目概览\n旧项目概览\n", "utf-8")

        tree = NavigationTree.model_validate({
            "generated_at": datetime.now(UTC).isoformat(),
            "project_summary": "项目围绕运行时组织。",
            "fallback_area_slug": "fallback",
            "source_digest": "seed",
            "areas": [
                {
                    "slug": "runtime",
                    "display_name": "运行时",
                    "summary": "运行时区域摘要",
                    "modules": [
                        {
                            "slug": "alpha",
                            "display_name": "Alpha",
                            "summary": "Alpha 摘要",
                            "module_type": "directory",
                            "include": ["src/**"],
                            "exclude": [],
                        }
                    ],
                    "recommended_order": ["alpha"],
                    "source_prefixes": ["src/"],
                    "is_fallback": False,
                },
                {
                    "slug": "fallback",
                    "display_name": "未分类（Fallback）",
                    "summary": "承接剩余内容",
                    "modules": [],
                    "recommended_order": [],
                    "source_prefixes": [],
                    "is_fallback": True,
                },
            ],
        })
        (annotations / "navigation_tree.json").write_text(
            __import__("json").dumps(tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        await save_index(
            IndexSnapshot(
                project_meta=ProjectMeta(
                    root_path=root,
                    created_at=datetime.now(UTC),
                    index_version="test",
                    file_count=1,
                    loc_total=2,
                ),
                entries=[
                    IndexEntry(
                        file_meta=FileMeta(
                            path=Path("src/alpha.py"),
                            language="python",
                            size_bytes=20,
                            mtime=datetime.now(UTC),
                            loc=2,
                        ),
                        symbols=[],
                        imports=[],
                        edges=[],
                    )
                ],
                created_at=datetime.now(UTC),
                build_stats=BuildStats(
                    total_files=1,
                    total_symbols=0,
                    total_edges=0,
                    duration_ms=1,
                    warnings=[],
                ),
            ),
            root_path=root,
        )
        await save_file_baseline(
            "src/alpha.py",
            content="def run():\n    return 1\n",
            content_hash="baseline-hash",
            symbols=[],
            root_path=root,
        )

        agent = DigestCleanupStageAgent(
            project_root=root,
            dirty_files=["src/alpha.py"],
            facts_text="cleanup facts",
            facts_truncated=False,
        )
        content = await _call_virtual(
            agent,
            "rewrite_sections",
            {
                "path": ".pce/annotations/modules/alpha.md",
                "sections": [{"heading": "核心职责", "content": ""}],
            },
        )
        assert "sections 已重写" in content
        rewritten = module_path.read_text("utf-8")
        assert "## 核心职责" in rewritten
        assert "- 旧职责" not in rewritten

        content = await _call_virtual(
            agent,
            "rewrite_sections",
            {
                "path": ".pce/annotations/modules/alpha.md",
                "sections": [{"heading": "#", "content": "新的模块概览"}],
            },
        )
        assert "sections 已重写" in content
        rewritten = module_path.read_text("utf-8")
        assert "新的模块概览" in rewritten

        content = await _call_virtual(
            agent,
            "rewrite_sections",
            {
                "path": ".pce/annotations/modules/alpha.md",
                "sections": [{"heading": "不存在的标题", "content": "x"}],
            },
        )
        assert "section 不存在" in content

        content = await _call_virtual(
            agent,
            "reset_annotation_to_skeleton",
            {"path": ".pce/annotations/modules/alpha.md"},
        )
        assert "已重置为骨架" in content
        reset_md = module_path.read_text("utf-8")
        assert "## 覆盖文件" in reset_md
        assert "## 核心职责" in reset_md
        assert "- src/alpha.py" in reset_md


async def _test_digest_gate_and_run_digest_support_cleanup_only() -> None:
    with TemporaryDirectory(prefix="pce-digest-cleanup-only-") as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "alpha.py").write_text("print('x')\n", "utf-8")
        insight_cache = InsightCache(root)
        dirty = DirtyState(changed=["src/alpha.py"], deleted=[])

        should_run, reason = await digest_agent.should_run_digest(
            project_root=root,
            insight_cache=insight_cache,
            dirty_state=dirty,
        )
        assert should_run is True
        assert reason == "dirty_files_require_cleanup"

        original_cleanup = digest_agent.run_digest_cleanup

        class _CleanupResult:
            def __init__(self, summary: str) -> None:
                self.summary = summary

        async def _fake_cleanup(**kwargs):
            return _CleanupResult("cleanup only path ok")

        digest_agent.run_digest_cleanup = _fake_cleanup
        try:
            result = await digest_agent.run_digest(
                project_root=root,
                serena_client=object(),
                insight_cache=insight_cache,
                dirty_state=dirty,
                skip_initial_sweep=True,
            )
        finally:
            digest_agent.run_digest_cleanup = original_cleanup

        assert result["executed"] is True
        assert result["resolved_tasks"] == 1
        assert "cleanup only path ok" in result["summary"]


def main() -> None:
    asyncio.run(_test_shared_budget_between_stage1_and_stage2())
    asyncio.run(_test_stage2_rejects_non_dirty_paths())
    asyncio.run(_test_stageC_rewrite_and_reset_tools())
    asyncio.run(_test_digest_gate_and_run_digest_support_cleanup_only())
    print("digest stage2/stageC tests passed")


if __name__ == "__main__":
    main()
