"""Digest router / planner 最小回归。"""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pce.digest_router import DigestPlannerV2
from pce.models import (
    AreaRecord,
    ChangedFileFact,
    InsightConfidence,
    InsightFact,
    ModuleDigestDelta,
    ModuleNavRecord,
    NavigationTree,
)
from pce.staging import DirtyState


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _delta(
    slug: str,
    *,
    path: str,
    area_slug: str,
    with_insight: bool = False,
) -> ModuleDigestDelta:
    insights = []
    if with_insight:
        insights.append(
            InsightFact(
                id=str(uuid.uuid4()),
                scope=path,
                content=f"{slug} insight",
                confidence=InsightConfidence.MEDIUM,
                created_at=datetime.now(UTC),
            )
        )
    return ModuleDigestDelta(
        module_id=str(uuid.uuid4()),
        module_slug=slug,
        module_name=slug,
        annotation_baseline="",
        related_insights=insights,
        changed_files=[
            ChangedFileFact(
                path=Path(path),
                status="modified",
            )
        ],
        change_scope_hint="module",
        external_context=[],
    )


def _tree() -> NavigationTree:
    now = datetime.now(UTC)
    return NavigationTree(
        generated_at=now,
        project_summary="demo",
        fallback_area_slug="fallback",
        source_digest="test",
        areas=[
            AreaRecord(
                slug="core",
                display_name="Core",
                summary="core area",
                recommended_order=["alpha", "beta"],
                source_prefixes=["src/core"],
                is_fallback=False,
                modules=[
                    ModuleNavRecord(
                        slug="alpha",
                        display_name="Alpha",
                        summary="alpha summary",
                        file_paths=["src/core/alpha.py"],
                    ),
                    ModuleNavRecord(
                        slug="beta",
                        display_name="Beta",
                        summary="beta summary",
                        file_paths=["src/core/beta.py"],
                    ),
                ],
            ),
            AreaRecord(
                slug="ui",
                display_name="UI",
                summary="ui area",
                recommended_order=["gamma"],
                source_prefixes=["src/ui"],
                is_fallback=False,
                modules=[
                    ModuleNavRecord(
                        slug="gamma",
                        display_name="Gamma",
                        summary="gamma summary",
                        file_paths=["src/ui/gamma.ts"],
                    )
                ],
            ),
            AreaRecord(
                slug="fallback",
                display_name="Fallback",
                summary="fallback",
                recommended_order=[],
                source_prefixes=[],
                is_fallback=True,
                modules=[],
            ),
        ],
    )


async def _run_case(deltas: list[ModuleDigestDelta], dirty: DirtyState) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nav_path = root / ".pce" / "annotations" / "navigation_tree.json"
        nav_path.parent.mkdir(parents=True, exist_ok=True)
        nav_path.write_text(
            json.dumps(_tree().model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )

        original = DigestPlannerV2.build

        planner = DigestPlannerV2(root, insight_cache=None)  # type: ignore[arg-type]

        from pce import digest_router as mod

        original_builder = mod.DigestDeltaBuilder.build_for_changes

        async def _fake_build_for_changes(self, *, changed_files, deleted_files=None):
            return deltas

        mod.DigestDeltaBuilder.build_for_changes = _fake_build_for_changes  # type: ignore[assignment]
        try:
            result = await planner.build(dirty)
        finally:
            mod.DigestDeltaBuilder.build_for_changes = original_builder  # type: ignore[assignment]
        return [item.to_summary_dict() for item in result.tasks]


async def main() -> None:
    single = await _run_case(
        [_delta("alpha", path="src/core/alpha.py", area_slug="core", with_insight=True)],
        DirtyState(changed=["src/core/alpha.py"], deleted=[]),
    )
    _assert(single[0]["task_level"] == "module", "单模块应归并为 module task")

    same_area = await _run_case(
        [
            _delta("alpha", path="src/core/alpha.py", area_slug="core"),
            _delta("beta", path="src/core/beta.py", area_slug="core"),
        ],
        DirtyState(changed=["src/core/alpha.py", "src/core/beta.py"], deleted=[]),
    )
    _assert(same_area[0]["task_level"] == "area", "同 area 多模块应归并为 area task")

    cross_area = await _run_case(
        [
            _delta("alpha", path="src/core/alpha.py", area_slug="core"),
            _delta("gamma", path="src/ui/gamma.ts", area_slug="ui"),
        ],
        DirtyState(changed=["src/core/alpha.py", "src/ui/gamma.ts"], deleted=[]),
    )
    _assert(cross_area[0]["task_level"] == "project", "跨 area 应归并为 project task")

    unresolved = await _run_case(
        [_delta("alpha", path="src/core/alpha.py", area_slug="core")],
        DirtyState(changed=["src/core/alpha.py", "node_modules/foo.js"], deleted=[]),
    )
    levels = [item["task_level"] for item in unresolved]
    _assert("unresolved" in levels, "未归属 dirty 文件应生成 unresolved task")

    print(json.dumps({"ok": True, "tests": [
        "single module -> module task",
        "same area modules -> area task",
        "cross area modules -> project task",
        "unowned dirty paths -> unresolved task",
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
