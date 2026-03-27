"""轻量 init helper 回归测试。

运行：
    uv run python scripts/test_topology_init_helpers.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pce.annotation_writer import (  # noqa: E402
    _build_section_slug_aliases,
    _build_sections_from_navigation_tree,
    _coerce_module_cognition_facts,
    _coerce_topology_navigation_tree,
    _enrich_tree_summaries_with_cognition,
    _inject_section_summaries_from_cognition,
    _remap_module_cognition_facts_model,
    _remap_navigation_tree_model_slugs,
    _render_area_md,
    _render_hierarchical_index_md,
    _render_module_md_from_cognition,
    _stabilize_sections_with_registry,
    _sync_tree_modules_from_sections,
)
from pce.models import (  # noqa: E402
    AreaRecord,
    FileMeta,
    IndexEntry,
    ModuleCognitionFact,
    NavigationTree,
)


def _entry(path: str, *, language: str = "python", loc: int = 10) -> IndexEntry:
    return IndexEntry(
        file_meta=FileMeta(
            path=Path(path),
            language=language,
            size_bytes=128,
            mtime=datetime.now(UTC),
            loc=loc,
        ),
        symbols=[],
        imports=[],
        edges=[],
    )


async def _assert_stabilize_preserves_source_sections(
    entries_map: dict[str, IndexEntry],
) -> None:
    source_sections = [
        {
            "slug": "agent-core-raw",
            "name": "Agent Core",
            "file_paths": ["pce/agent.py", "pce/base_agent.py"],
            "body_lines": [
                "文件：pce/agent.py, pce/base_agent.py",
                "职责：负责主 Agent 循环与基础抽象。",
                "详细认知：.pce/annotations/modules/agent-core-raw.md",
            ],
        }
    ]

    with TemporaryDirectory(prefix="pce-topology-test-") as tmpdir:
        stabilized = await _stabilize_sections_with_registry(
            source_sections,
            root_path=Path(tmpdir),
            entries_map=entries_map,
        )

    assert source_sections[0]["slug"] == "agent-core-raw"
    assert source_sections[0]["name"] == "Agent Core"
    assert stabilized[0]["slug"] == "agent-core"
    assert stabilized[0]["name"] == "Agent Core"
    assert _build_section_slug_aliases(source_sections, stabilized) == {
        "agent-core-raw": "agent-core"
    }


def main() -> None:
    entries_map = {
        "pce/agent.py": _entry("pce/agent.py", loc=200),
        "pce/base_agent.py": _entry("pce/base_agent.py", loc=150),
    }
    asyncio.run(_assert_stabilize_preserves_source_sections(entries_map))

    raw_tree = _coerce_topology_navigation_tree(
        {
            "project_summary": "项目围绕 agent 运行时组织。",
            "fallback_area_slug": "fallback",
            "areas": [
                {
                    "slug": "runtime",
                    "display_name": "运行时",
                    "summary": "负责 Agent 主循环与工具协作。",
                    "modules": [
                        {
                            "slug": "agent-core-raw",
                            "display_name": "Agent Core",
                            "summary": "负责主 Agent 循环与基础抽象。",
                            "file_paths": ["pce/agent.py", "pce/base_agent.py"],
                        }
                    ],
                    "recommended_order": ["agent-core-raw"],
                    "source_prefixes": ["pce/"],
                    "is_fallback": False,
                },
                {
                    "slug": "fallback",
                    "display_name": "未分类（Fallback）",
                    "summary": "承接零散模块。",
                    "modules": [],
                    "recommended_order": [],
                    "source_prefixes": [],
                    "is_fallback": True,
                },
            ],
        },
        facts={"source_digest": "abc123"},
        active_slugs=None,
    )
    assert raw_tree.areas[0].module_slugs == ["agent-core-raw"]
    assert raw_tree.areas[0].modules[0].display_name == "Agent Core"

    raw_sections = _build_sections_from_navigation_tree(raw_tree)
    assert len(raw_sections) == 1
    assert raw_sections[0]["slug"] == "agent-core-raw"
    assert any(line.startswith("职责：") for line in raw_sections[0]["body_lines"])

    stable_sections = [
        {
            "slug": "agent-core",
            "name": "Agent Core",
            "file_paths": ["pce/agent.py", "pce/base_agent.py"],
            "body_lines": [
                "文件：pce/agent.py, pce/base_agent.py",
                "职责：负责主 Agent 循环与基础抽象。",
                "详细认知：.pce/annotations/modules/agent-core.md",
            ],
        }
    ]
    slug_aliases = _build_section_slug_aliases(raw_sections, stable_sections)
    assert slug_aliases == {"agent-core-raw": "agent-core"}

    stable_tree = _remap_navigation_tree_model_slugs(
        raw_tree,
        slug_aliases,
        {section["slug"]: section for section in stable_sections},
    )
    stable_tree = _sync_tree_modules_from_sections(
        stable_tree,
        {section["slug"]: section for section in stable_sections},
    )
    assert stable_tree.areas[0].module_slugs == ["agent-core"]
    assert stable_tree.areas[0].modules[0].slug == "agent-core"

    cognition = _coerce_module_cognition_facts(
        {
            "modules": [
                {
                    "slug": "agent-core-raw",
                    "display_name": "Agent Core",
                    "core_responsibility": ["负责主 Agent 循环。"],
                    "key_flow": [],
                    "external_collaboration": ["与 Serena 工具层协作。"],
                    "risks_constraints": [],
                    "key_anchors": [],
                }
            ]
        },
        expected_slugs={"agent-core-raw"},
    )
    remapped_cognition = _remap_module_cognition_facts_model(
        cognition,
        slug_aliases,
        {section["slug"]: section for section in stable_sections},
    )
    assert [fact.slug for fact in remapped_cognition.modules] == ["agent-core"]

    module_md = _render_module_md_from_cognition(
        "Agent Core",
        list(entries_map.values()),
        ModuleCognitionFact.model_validate(remapped_cognition.modules[0].model_dump()),
    )
    assert "## 覆盖文件" in module_md
    assert "## 核心职责" in module_md
    assert "## 外部协作" in module_md
    assert "## 关键符号" not in module_md

    injected_sections = _inject_section_summaries_from_cognition(
        stable_sections,
        remapped_cognition,
    )
    assert any(line.startswith("职责：") for line in injected_sections[0]["body_lines"])
    sections_by_slug = {section["slug"]: section for section in injected_sections}

    enriched_tree = _enrich_tree_summaries_with_cognition(
        NavigationTree(
            generated_at=datetime.now(UTC),
            project_summary="项目当前共 1 个模块。",
            fallback_area_slug="fallback",
            source_digest="abc123",
            areas=[
                AreaRecord(
                    slug="runtime",
                    display_name="运行时",
                    summary="主要覆盖 `pce/` 前缀下的模块。",
                    modules=[
                        {
                            "slug": "agent-core",
                            "display_name": "Agent Core",
                            "summary": "负责主 Agent 循环与基础抽象。",
                            "file_paths": ["pce/agent.py", "pce/base_agent.py"],
                        }
                    ],
                    module_slugs=["agent-core"],
                    recommended_order=["agent-core"],
                    source_prefixes=["pce/"],
                    is_fallback=False,
                ),
                AreaRecord(
                    slug="fallback",
                    display_name="未分类（Fallback）",
                    summary="承接零散模块。",
                    modules=[],
                    module_slugs=[],
                    recommended_order=[],
                    source_prefixes=[],
                    is_fallback=True,
                ),
            ],
        ),
        sections_by_slug=sections_by_slug,
        cognition_facts=remapped_cognition,
    )
    assert "项目当前共" not in enriched_tree.project_summary
    assert "承接零散模块" not in enriched_tree.project_summary
    assert "主要覆盖 `" not in enriched_tree.areas[0].summary

    index_md = _render_hierarchical_index_md(enriched_tree, sections_by_slug)
    assert "## 项目概览" in index_md
    assert "## 区域入口" in index_md
    assert "[运行时](areas/runtime.md)" in index_md
    assert "Agent Core" in index_md

    area_md = _render_area_md(enriched_tree.areas[0], sections_by_slug)
    assert "## 区域说明" in area_md
    assert "## 模块列表" in area_md
    assert "[Agent Core](../modules/agent-core.md)" in area_md
    assert "负责主 Agent 循环。" in area_md
    assert "关键流程" not in area_md

    print("ok: topology init helpers")


if __name__ == "__main__":
    main()
