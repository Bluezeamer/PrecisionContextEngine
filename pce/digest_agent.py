"""Digest 两阶段轻量入口。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .digest_cognition_agent import run_digest_assimilation, run_digest_filter
from .digest_delta_builder import DigestDeltaBuilder
from .file_discovery import filter_visible_paths
from .insight_cache import InsightCache
from .models import ChangedFileFact, InsightFact
from .serena_client import SerenaClient
from .staging import DirtyState

import logging

logger = logging.getLogger(__name__)

_STAGE_A_MAX_FACTS_CHARS = 32000


@dataclass
class InsightBatch:
    insights: list[InsightFact]


def _estimate_insight_chars(insight: InsightFact) -> int:
    return len(insight.id) + len(insight.scope) + len(insight.content) + 64


def _chunk_insights(insights: list[InsightFact], *, max_chars: int = _STAGE_A_MAX_FACTS_CHARS) -> list[InsightBatch]:
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


def _dedupe_changed_file_facts(deltas: list[Any]) -> list[ChangedFileFact]:
    deduped: list[ChangedFileFact] = []
    seen: set[str] = set()
    for delta in deltas:
        for file_fact in delta.changed_files:
            rel = str(file_fact.path)
            if rel in seen:
                continue
            seen.add(rel)
            deduped.append(file_fact)
    return deduped


async def _load_active_insights(insight_cache: InsightCache) -> list[InsightFact]:
    records = await insight_cache.get_all_records(include_stale=False)
    result: list[InsightFact] = []
    for record in records:
        content = await insight_cache.get_entry_content(record.id)
        if not content:
            continue
        result.append(
            InsightFact(
                id=record.id,
                scope=record.scope,
                content=content,
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
    if not insights:
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
            "executed": False,
            "summary": "",
            "resolved_tasks": 0,
            "pending_tasks": 0,
            "deleted_insights": 0,
            "warnings": [],
        }

    visible_changed = filter_visible_paths(project_root, dirty_state.changed)
    visible_deleted = filter_visible_paths(project_root, dirty_state.deleted)
    dirty_files = list(dict.fromkeys([*visible_changed, *visible_deleted]))
    patch_builder = DigestDeltaBuilder(project_root, insight_cache)
    deltas, _ = await patch_builder.build_for_insights(
        changed_files=visible_changed,
        deleted_files=visible_deleted,
    )
    patch_facts = _dedupe_changed_file_facts(deltas)

    filter_start = time.monotonic()
    summaries: list[str] = []
    warnings: list[str] = []
    deleted_insight_ids: list[str] = []
    resolved_batches = 0
    pending_batches = 0
    for batch_index, batch in enumerate(_chunk_insights(insights), start=1):
        logger.info("Digest stageA start: batch=%d insights=%d", batch_index, len(batch.insights))
        try:
            decision = await run_digest_filter(
                project_root=project_root,
                insights=batch.insights,
                model=model,
                provider=provider,
                serena_client=serena_client,
            )
        except Exception as exc:
            pending_batches += 1
            warnings.append(f"Digest stageA 失败: batch={batch_index}: {exc}")
            continue

        batch_ids = [item.id for item in batch.insights]
        drop_ids = [item for item in decision.drop_insight_ids if item in batch_ids]
        keep_ids = [item for item in decision.keep_insight_ids if item in batch_ids and item not in drop_ids]
        unresolved_ids = [item for item in batch_ids if item not in drop_ids and item not in keep_ids]
        # 默认保守：未明确 keep 的视作 drop，避免 insight 空挂
        drop_ids.extend(unresolved_ids)
        drop_ids = list(dict.fromkeys(drop_ids))
        kept_insights = [item for item in batch.insights if item.id in keep_ids]

        if drop_ids:
            await insight_cache.delete_by_ids(drop_ids)
            deleted_insight_ids.extend(drop_ids)

        if not kept_insights:
            resolved_batches += 1
            notes = "；".join(decision.notes) if decision.notes else "全部筛除"
            summaries.append(f"[batch:{batch_index}] stageA complete: {notes}")
            continue

        logger.info("Digest stageB start: batch=%d insights=%d", batch_index, len(kept_insights))
        try:
            result = await run_digest_assimilation(
                project_root=project_root,
                insights=kept_insights,
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

        handled_ids = [item.id for item in kept_insights]
        await insight_cache.delete_by_ids(handled_ids)
        deleted_insight_ids.extend(handled_ids)
        resolved_batches += 1
        summaries.append(f"[batch:{batch_index}] {result.summary}".strip())

    logger.info("Digest 阶段耗时: run_stageA_stageB=%.2fs", time.monotonic() - filter_start)

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
        "executed": True,
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
    del project_root, dirty_state
    sweep_start = time.monotonic()
    try:
        await insight_cache.sweep_stale()
    except Exception as exc:
        logger.warning("digest gate sweep_stale 失败，继续检查 insights: %s", exc)
    finally:
        logger.info("Digest gate 阶段耗时: sweep_stale=%.2fs", time.monotonic() - sweep_start)
    records = await insight_cache.get_all_records(include_stale=False)
    return (True, "actionable_fresh_insights") if records else (False, "no_actionable_insights")
