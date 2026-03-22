"""
Digest 真实场景验证脚本。

用途：
1. 查看当前项目 dirty state / insight / DigestPlanner 任务规划；
2. 可选执行一次真实 run_digest，直接观察 resolved/pending/warnings；
3. 为 hicolors 等复杂项目提供专门的 Digest 验证入口，而不是混在 query/impact e2e 中。

运行：
    uv run python scripts/test_digest_runtime.py
    uv run python scripts/test_digest_runtime.py --execute
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pce._env import configure_litellm_runtime, get_completion_overrides, get_env_text
from pce.digest_agent import DigestPlanner, run_digest
from pce.insight_cache import InsightCache
from pce.serena_client import SerenaClient
from pce.staging import StagingArea

logger = logging.getLogger("pce.digest-runtime")


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digest 真实场景验证脚本")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="在规划完成后，继续执行一次真实 run_digest",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=8,
        help="输出的任务预览数量上限",
    )
    return parser.parse_args()


async def _count_baselines(project_root: Path) -> int:
    baselines_dir = project_root / ".pce" / "baselines" / "files"
    if not baselines_dir.exists():
        return 0
    return len(list(baselines_dir.rglob("*.json")))


async def _count_annotation_modules(project_root: Path) -> int:
    modules_dir = project_root / ".pce" / "annotations" / "modules"
    if not modules_dir.exists():
        return 0
    return len(list(modules_dir.glob("*.md")))


async def main() -> None:
    args = _parse_args()
    root = _root()
    load_dotenv(root / ".env")
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or root).resolve()
    staging = StagingArea(project_path)
    insight_cache = InsightCache(project_path)

    dirty = await staging.list_pending_reindex()
    insight_stats = await insight_cache.stats()
    planner = DigestPlanner(project_root=project_path, insight_cache=insight_cache)
    task_list = await planner.build(dirty)
    summary = task_list.to_summary_dict()

    payload: dict[str, Any] = {
        "project_path": str(project_path),
        "dirty_state": {
            "changed_count": len(dirty.changed),
            "deleted_count": len(dirty.deleted),
            "changed_preview": dirty.changed[: args.max_items],
            "deleted_preview": dirty.deleted[: args.max_items],
        },
        "insight_stats": insight_stats.model_dump(mode="json"),
        "annotation_modules_count": await _count_annotation_modules(project_path),
        "baseline_file_count": await _count_baselines(project_path),
        "planner": {
            "warnings": list(task_list.warnings),
            "task_count": len(task_list.items),
            "task_preview": summary["items"][: args.max_items],
        },
    }

    if args.execute:
        provider = get_env_text("PCE_PROVIDER")
        model = get_env_text("PCE_MODEL")
        overrides = get_completion_overrides()
        if not provider or not model or not overrides.get("api_key"):
            raise RuntimeError(
                "执行 run_digest 需要已配置 PCE_PROVIDER / PCE_MODEL / PCE_API_KEY"
            )

        async with SerenaClient.create(project_path) as serena_client:
            digest_result = await run_digest(
                project_root=project_path,
                serena_client=serena_client,
                insight_cache=insight_cache,
                dirty_state=dirty,
                model=model,
                provider=provider,
            )
        payload["digest_result"] = digest_result
        payload["baseline_file_count_after"] = await _count_baselines(project_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
