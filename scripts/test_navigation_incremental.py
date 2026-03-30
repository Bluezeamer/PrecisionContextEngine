"""增量导航链路最小回归测试。

运行：
    uv run python scripts/test_navigation_incremental.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pce.annotation_writer as annotation_writer  # noqa: E402
from pce.annotation_writer import (  # noqa: E402
    _navigation_tree_path,
    _run_incremental_navigation_update,
    _update_annotations_incremental,
)
from pce.models import (  # noqa: E402
    AreaRecord,
    FileMeta,
    IndexEntry,
    NavigationTree,
    ProjectMeta,
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


def _project_meta(root: Path, entries: list[IndexEntry]) -> ProjectMeta:
    return ProjectMeta(
        root_path=root,
        created_at=datetime.now(UTC),
        index_version="test",
        file_count=len(entries),
        loc_total=sum(entry.file_meta.loc for entry in entries),
    )


def _build_tree() -> NavigationTree:
    return NavigationTree(
        generated_at=datetime.now(UTC),
        project_summary="测试项目",
        fallback_area_slug="fallback",
        source_digest="seed",
        areas=[
            AreaRecord(
                slug="runtime",
                display_name="运行时",
                summary="运行时能力",
                modules=[
                    {
                        "slug": "alpha",
                        "display_name": "Alpha",
                        "summary": "Alpha 模块",
                        "module_type": "directory",
                        "include": ["src/alpha/**"],
                        "exclude": [],
                    },
                    {
                        "slug": "legacy-single",
                        "display_name": "Legacy Single",
                        "summary": "遗留单文件模块",
                        "module_type": "file_centered",
                        "include": ["src/legacy.py"],
                        "exclude": [],
                    },
                ],
                module_slugs=["alpha", "legacy-single"],
                recommended_order=["alpha", "legacy-single"],
                source_prefixes=["src/"],
                is_fallback=False,
            ),
            AreaRecord(
                slug="fallback",
                display_name="未分类（Fallback）",
                summary="承接剩余内容",
                modules=[
                    {
                        "slug": "fallback-residual",
                        "display_name": "Fallback Residual",
                        "summary": "剩余文件",
                        "module_type": "residual",
                        "include": [],
                        "exclude": [],
                    }
                ],
                module_slugs=["fallback-residual"],
                recommended_order=["fallback-residual"],
                source_prefixes=[],
                is_fallback=True,
            ),
        ],
    )


async def _test_created_file_already_covered_returns_no_change() -> None:
    with TemporaryDirectory(prefix="pce-nav-incremental-covered-") as tmpdir:
        root = Path(tmpdir)
        tree = _build_tree()
        entries = [
            _entry("src/alpha/a.py"),
            _entry("src/alpha/new_file.py"),
        ]

        result = await _run_incremental_navigation_update(
            entries=entries,
            project_meta=_project_meta(root, entries),
            root_path=root,
            tree=tree,
            changed_files=["src/alpha/new_file.py"],
            deleted_files=[],
            model=None,
            serena_client=object(),  # 不会实际触发 agent
        )

        assert result["decision"] == "no_change"
        assert result["rationale"] == "created_files_already_covered"


async def _test_module_update_persists_stable_tree() -> None:
    with TemporaryDirectory(prefix="pce-nav-incremental-module-") as tmpdir:
        root = Path(tmpdir)
        modules_dir = root / ".pce" / "annotations" / "modules"
        modules_dir.mkdir(parents=True, exist_ok=True)

        current_tree = _build_tree()
        (_navigation_tree_path(root)).write_text(
            json.dumps(current_tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )
        (root / ".pce" / "annotations" / "index.md").write_text("# 项目认知导航\n", "utf-8")
        (modules_dir / "alpha.md").write_text(
            "\n".join([
                "# Alpha",
                "",
                "## 覆盖文件",
                "- src/alpha/a.py",
                "",
                "## 核心职责",
                "- 保留原有正文",
                "",
            ]) + "\n",
            "utf-8",
        )
        (modules_dir / "legacy-single.md").write_text(
            "\n".join([
                "# Legacy Single",
                "",
                "## 覆盖文件",
                "- src/legacy.py",
                "",
                "## 核心职责",
                "- 该模块后续应被清理",
                "",
            ]) + "\n",
            "utf-8",
        )

        entries = [
            _entry("src/alpha/a.py"),
            _entry("src/beta/new.py"),
        ]

        candidate_tree = NavigationTree(
            generated_at=datetime.now(UTC),
            project_summary="测试项目",
            fallback_area_slug="fallback",
            source_digest="candidate",
            areas=[
                AreaRecord(
                    slug="runtime",
                    display_name="运行时",
                    summary="运行时能力",
                    modules=[
                        {
                            "slug": "alpha",
                            "display_name": "Alpha",
                            "summary": "Alpha 模块",
                            "module_type": "directory",
                            "include": ["src/alpha/**"],
                            "exclude": [],
                        },
                        {
                            "slug": "beta",
                            "display_name": "Beta",
                            "summary": "Beta 模块",
                            "module_type": "directory",
                            "include": ["src/beta/**"],
                            "exclude": [],
                        },
                    ],
                    module_slugs=["alpha", "beta"],
                    recommended_order=["alpha", "beta"],
                    source_prefixes=["src/"],
                    is_fallback=False,
                ),
                AreaRecord(
                    slug="fallback",
                    display_name="未分类（Fallback）",
                    summary="承接剩余内容",
                    modules=[],
                    module_slugs=[],
                    recommended_order=[],
                    source_prefixes=[],
                    is_fallback=True,
                ),
            ],
        )

        original_runner = annotation_writer._run_incremental_navigation_update
        original_full_write = annotation_writer._write_annotations

        async def _fake_runner(**_: object) -> dict[str, object]:
            return {
                "decision": "module_update",
                "rationale": "test",
                "navigation_tree": candidate_tree,
            }

        async def _unexpected_full_write(*args: object, **kwargs: object) -> None:
            raise AssertionError("本测试不应回落到全量重建")

        annotation_writer._run_incremental_navigation_update = _fake_runner
        annotation_writer._write_annotations = _unexpected_full_write
        try:
            await _update_annotations_incremental(
                entries,
                _project_meta(root, entries),
                root,
                changed_files=["src/beta/new.py"],
                deleted_files=["src/legacy.py"],
                model=None,
                serena_client=object(),
            )
        finally:
            annotation_writer._run_incremental_navigation_update = original_runner
            annotation_writer._write_annotations = original_full_write

        persisted_tree = json.loads(_navigation_tree_path(root).read_text("utf-8"))
        runtime_area = next(area for area in persisted_tree["areas"] if area["slug"] == "runtime")
        assert runtime_area["module_slugs"] == ["alpha", "beta"]
        assert persisted_tree["source_digest"]

        alpha_md = (modules_dir / "alpha.md").read_text("utf-8")
        beta_md = (modules_dir / "beta.md").read_text("utf-8")
        assert "保留原有正文" in alpha_md
        assert "## 覆盖文件" in beta_md
        assert "- src/beta/new.py" in beta_md
        assert not (modules_dir / "legacy-single.md").exists()


def main() -> None:
    asyncio.run(_test_created_file_already_covered_returns_no_change())
    asyncio.run(_test_module_update_persists_stable_tree())
    print("incremental navigation tests passed")


if __name__ == "__main__":
    main()
