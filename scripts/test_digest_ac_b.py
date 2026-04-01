"""Digest `AC -> B` 最小回归测试。

运行：
    uv run python scripts/test_digest_ac_b.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from pce.digest_cognition_agent import DigestAssimilationStageAgent, DigestAuditStageAgent
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
from pce.memory import save_file_baseline, save_index


def _insight(idx: int, *, question: str, answer: str) -> InsightFact:
    return InsightFact(
        id=f"00000000-0000-0000-0000-{idx:012d}",
        question=question,
        answer=answer,
        confidence=InsightConfidence.MEDIUM,
        created_at=datetime.now(UTC),
    )


async def _call_virtual(agent, name: str, args: dict) -> str:
    tool_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
    result = await agent.handle_virtual_tool(tool_call, state=None)  # type: ignore[arg-type]
    assert result is not None
    return str(result.get("content") or "")


async def _prepare_fixture_root() -> tuple[TemporaryDirectory[str], Path, Path]:
    tmpdir = TemporaryDirectory(prefix="pce-digest-acb-")
    root = Path(tmpdir.name)
    annotations = root / ".pce" / "annotations"
    modules_dir = annotations / "modules"
    areas_dir = annotations / "areas"
    modules_dir.mkdir(parents=True, exist_ok=True)
    areas_dir.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "alpha.py").write_text("def run():\n    return 2\n", "utf-8")

    module_path = modules_dir / "alpha.md"
    module_path.write_text(
        "# Alpha\n\n模块概览旧内容\n\n## 覆盖文件\n- src/alpha.py\n\n## 核心职责\n- 旧职责\n\n## 关键流程\n- 旧流程\n",
        "utf-8",
    )
    (areas_dir / "runtime.md").write_text("# 运行时\n\n## 区域说明\n旧区域说明\n", "utf-8")
    (annotations / "index.md").write_text("# 项目认知导航\n\n旧项目概览\n\n## 项目概览\n旧项目概览 section\n", "utf-8")

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
        json.dumps(tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
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
    return tmpdir, root, module_path


async def _test_stage_ac_tools() -> None:
    tmpdir, root, module_path = await _prepare_fixture_root()
    try:
        agent = DigestAuditStageAgent(
            project_root=root,
            insights=[_insight(1, question="alpha 做什么", answer="alpha 负责执行 run")],
            dirty_files=["src/alpha.py"],
            facts_text="audit facts",
            facts_truncated=False,
        )

        content = await _call_virtual(agent, "read_diff", {"paths": ["src/alpha.py"]})
        assert "a/src/alpha.py" in content
        assert "b/src/alpha.py" in content

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
            "reset_annotation_to_skeleton",
            {"path": ".pce/annotations/modules/alpha.md"},
        )
        assert "已重置为骨架" in content
        reset_md = module_path.read_text("utf-8")
        assert "## 覆盖文件" in reset_md
        assert "## 核心职责" in reset_md
        assert "- src/alpha.py" in reset_md
    finally:
        tmpdir.cleanup()


async def _test_stage_b_rewrite_only() -> None:
    tmpdir, root, module_path = await _prepare_fixture_root()
    try:
        agent = DigestAssimilationStageAgent(
            project_root=root,
            insight_count=1,
            facts_text="assimilation facts",
            facts_truncated=False,
        )

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
    finally:
        tmpdir.cleanup()


def main() -> None:
    asyncio.run(_test_stage_ac_tools())
    asyncio.run(_test_stage_b_rewrite_only())
    print("digest AC/B tests passed")


if __name__ == "__main__":
    main()
