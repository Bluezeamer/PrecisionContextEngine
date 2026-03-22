from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pce.digest_delta_builder import DigestDeltaBuilder
from pce.insight_cache import InsightCache
from pce.memory import save_file_baseline, save_index
from pce.models import BuildStats, FileMeta, IndexEntry, IndexSnapshot, ProjectMeta
from pce.module_registry import ModuleRegistryManager


async def _main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "current").mkdir(parents=True, exist_ok=True)
        (root / "modern_v2").mkdir(parents=True, exist_ok=True)
        (root / "current" / "live.py").write_text("def live():\n    return 1\n", "utf-8")
        (root / "modern_v2" / "models.py").write_text("class Model:\n    pass\n", "utf-8")

        manager = ModuleRegistryManager(root)
        first = await manager.get_or_create_module(
            display_name="Legacy Module",
            file_paths=["legacy/old.py", "current/live.py"],
            key_symbols=["live"],
        )
        second = await manager.get_or_create_module(
            display_name="Legacy Module",
            file_paths=["current/live.py"],
            key_symbols=["live"],
        )
        modern = await manager.get_or_create_module(
            display_name="Modern Module",
            file_paths=["modern_v2/models.py"],
            key_symbols=["Model"],
        )

        assert first.module_id == second.module_id
        assert second.file_paths == ["current/live.py"]
        assert "legacy/old.py" in second.historical_file_paths

        now = datetime.now(UTC)
        snapshot = IndexSnapshot(
            project_meta=ProjectMeta(
                root_path=root,
                created_at=now,
                index_version="1",
                file_count=1,
                loc_total=2,
            ),
            entries=[
                IndexEntry(
                    file_meta=FileMeta(
                        path=Path("current/live.py"),
                        language="python",
                        size_bytes=(root / "current" / "live.py").stat().st_size,
                        mtime=now,
                        loc=2,
                    ),
                    symbols=[],
                    imports=[],
                    edges=[],
                ),
                IndexEntry(
                    file_meta=FileMeta(
                        path=Path("modern_v2/models.py"),
                        language="python",
                        size_bytes=(root / "modern_v2" / "models.py").stat().st_size,
                        mtime=now,
                        loc=2,
                    ),
                    symbols=[],
                    imports=[],
                    edges=[],
                ),
            ],
            created_at=now,
            build_stats=BuildStats(
                total_files=2,
                total_symbols=0,
                total_edges=0,
                duration_ms=1,
                warnings=[],
            ),
        )
        await save_index(snapshot, root_path=root)
        await save_file_baseline(
            "legacy/old.py",
            content="def old():\n    return 0\n",
            content_hash="legacy-hash",
            symbols=[],
            root_path=root,
        )
        await save_file_baseline(
            "legacy_pkg/models.py",
            content="class OldModel:\n    pass\n",
            content_hash="legacy-model-hash",
            symbols=[],
            root_path=root,
        )

        builder = DigestDeltaBuilder(root, InsightCache(root))
        deltas = await builder.build_for_changes(
            changed_files=[],
            deleted_files=["legacy/old.py", "legacy_pkg/models.py"],
        )

        delta_map = {delta.module_slug: delta for delta in deltas}
        assert second.slug in delta_map
        assert len(delta_map[second.slug].changed_files) == 1
        assert str(delta_map[second.slug].changed_files[0].path) == "legacy/old.py"
        assert delta_map[second.slug].changed_files[0].status == "deleted"
        assert delta_map[second.slug].changed_files[0].old_content == "def old():\n    return 0\n"

        modern_delta = next(
            delta for delta in deltas if any(str(f.path) == "legacy_pkg/models.py" for f in delta.changed_files)
        )
        assert modern_delta.module_slug == modern.slug

    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "module_registry 保留历史文件归属",
                    "删除文件可通过历史归属命中原模块",
                    "deleted file 能进入 digest delta 并携带旧 baseline",
                    "历史归属缺失时，deleted path 可通过轻量路径相似性回挂模块",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())
