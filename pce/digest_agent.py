"""Digest 两阶段轻量入口。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .digest_cognition_agent import (
    run_digest_audit,
    run_digest_assimilation,
)
from .digest_delta_builder import DigestDeltaBuilder
from .file_discovery import filter_visible_paths
from .insight_cache import InsightCache
from .models import InsightFact
from .serena_client import SerenaClient
from .staging import DirtyState

import logging

logger = logging.getLogger(__name__)

_STAGE_B_MAX_FACTS_CHARS = 32000


@dataclass
class InsightBatch:
    insights: list[InsightFact]


def _estimate_insight_chars(insight: InsightFact) -> int:
    return len(insight.id) + len(insight.question) + len(insight.answer) + 64


def _chunk_insights(insights: list[InsightFact], *, max_chars: int = _STAGE_B_MAX_FACTS_CHARS) -> list[InsightBatch]:
    if not insights:
        return []
    batches: list[InsightBatch] = []
    current: list[InsightFact] = []
    used = 0
    for insight in insights:
        size = _estimate_insight_chars(insight)
        if current and used + size > max_chars:
            batches.append(InsightBatch(insights=list(current)))
            current = []
            used = 0
        current.append(insight)
        used += size
    if current:
        batches.append(InsightBatch(insights=list(current)))
    return batches


async def _load_active_insights(insight_cache: InsightCache) -> list[InsightFact]:
    records = await insight_cache.get_all_records(include_stale=False)
    result: list[InsightFact] = []
    for record in records:
        entry = await insight_cache.get_entry(record.id)
        if entry is None:
            continue
        result.append(
            InsightFact(
                id=record.id,
                question=entry.question,
                answer=entry.answer,
                confidence=record.confidence,
                created_at=record.created_at,
            )
        )
    return result


async def run_digest(
    *,
    project_root: Path,
    serena_client: SerenaClient,
    insight_cache: InsightCache,
    dirty_state: DirtyState,
    model: str | None = None,
    provider: str | None = None,
    skip_initial_sweep: bool = False,
) -> dict[str, Any]:
    digest_start = time.monotonic()
    visible_changed = filter_visible_paths(project_root, dirty_state.changed)
    visible_deleted = filter_visible_paths(project_root, dirty_state.deleted)
    dirty_files = list(dict.fromkeys([*visible_changed, *visible_deleted]))
    if not skip_initial_sweep:
        sweep_start = time.monotonic()
        try:
            await insight_cache.sweep_stale()
        except Exception as exc:
            logger.warning("Digest 前 sweep_stale 失败（已忽略）: %s", exc)
        finally:
            logger.info("Digest 阶段耗时: sweep_stale=%.2fs", time.monotonic() - sweep_start)

    load_start = time.monotonic()
    insights = await _load_active_insights(insight_cache)
    logger.info(
        "Digest 阶段耗时: load_insights=%.2fs (insights=%d)",
        time.monotonic() - load_start,
        len(insights),
    )
    patch_builder = DigestDeltaBuilder(project_root)
    patch_facts = await patch_builder.build_patch_facts(
        changed_files=visible_changed,
        deleted_files=visible_deleted,
    )
    filter_start = time.monotonic()
    summaries: list[str] = []
    warnings: list[str] = []
    deleted_insight_ids: list[str] = []
    resolved_batches = 0
    pending_batches = 0

    kept_insights: list[InsightFact] = []
    if insights or dirty_files:
        logger.info("Digest stageAC start: insights=%d dirty_files=%d", len(insights), len(dirty_files))
        try:
            audit_result = await run_digest_audit(
                project_root=project_root,
                insights=insights,
                dirty_files=dirty_files,
                patch_facts=patch_facts,
                model=model,
                provider=provider,
                serena_client=serena_client,
            )
        except Exception as exc:
            pending_batches += 1
            warnings.append(f"Digest stageAC 失败: {exc}")
            audit_result = None
        if audit_result is not None:
            insight_ids = [item.id for item in insights]
            drop_ids = [item for item in audit_result.drop_insight_ids if item in insight_ids]
            keep_ids = [item for item in audit_result.keep_insight_ids if item in insight_ids and item not in drop_ids]
            unresolved_ids = [item for item in insight_ids if item not in drop_ids and item not in keep_ids]
            drop_ids.extend(unresolved_ids)
            drop_ids = list(dict.fromkeys(drop_ids))
            kept_insights = [item for item in insights if item.id in keep_ids]
            if drop_ids:
                await insight_cache.delete_by_ids(drop_ids)
                deleted_insight_ids.extend(drop_ids)
            resolved_batches += 1
            if audit_result.summary:
                summaries.append(f"[audit] {audit_result.summary}".strip())
            elif audit_result.notes:
                summaries.append(f"[audit] {'；'.join(audit_result.notes)}")

    for batch_index, batch in enumerate(_chunk_insights(kept_insights), start=1):
        logger.info("Digest stageB start: batch=%d insights=%d", batch_index, len(batch.insights))
        try:
            result = await run_digest_assimilation(
                project_root=project_root,
                insights=batch.insights,
                dirty_files=dirty_files,
                patch_facts=patch_facts,
                model=model,
                provider=provider,
                serena_client=serena_client,
            )
        except Exception as exc:
            pending_batches += 1
            warnings.append(f"Digest stageB 失败: batch={batch_index}: {exc}")
            continue

        handled_ids = [item.id for item in batch.insights]
        await insight_cache.delete_by_ids(handled_ids)
        deleted_insight_ids.extend(handled_ids)
        resolved_batches += 1
        summaries.append(f"[batch:{batch_index}] {result.summary}".strip())

    logger.info("Digest 阶段耗时: run_stageAC_stageB=%.2fs", time.monotonic() - filter_start)

    cleanup_start = time.monotonic()
    try:
        removed = await insight_cache.cleanup_stale()
        if removed:
            logger.info("Digest cleanup_stale: 删除 %d 条", removed)
    except Exception as exc:
        logger.warning("Digest cleanup_stale 失败（已忽略）: %s", exc)
    finally:
        logger.info("Digest 阶段耗时: cleanup_stale=%.2fs", time.monotonic() - cleanup_start)

    logger.info("Digest 总耗时: %.2fs", time.monotonic() - digest_start)
    return {
        "executed": bool(insights or dirty_files),
        "summary": "\n\n".join(item for item in summaries if item).strip(),
        "resolved_tasks": resolved_batches,
        "pending_tasks": pending_batches,
        "deleted_insights": len(set(deleted_insight_ids)),
        "warnings": warnings,
    }


async def should_run_digest(
    *,
    project_root: Path,
    insight_cache: InsightCache,
    dirty_state: DirtyState,
) -> tuple[bool, str]:
    sweep_start = time.monotonic()
    try:
        await insight_cache.sweep_stale()
    except Exception as exc:
        logger.warning("digest gate sweep_stale 失败，继续检查 insights: %s", exc)
    finally:
        logger.info("Digest gate 阶段耗时: sweep_stale=%.2fs", time.monotonic() - sweep_start)
    records = await insight_cache.get_all_records(include_stale=False)
    if records:
        return True, "actionable_fresh_insights"
    visible_changed = filter_visible_paths(project_root, dirty_state.changed)
    visible_deleted = filter_visible_paths(project_root, dirty_state.deleted)
    if visible_changed or visible_deleted:
        return True, "dirty_files_require_cleanup"
    return False, "no_actionable_insights_or_dirty_files"
