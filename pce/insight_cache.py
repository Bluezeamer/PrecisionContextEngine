"""Insight Cache 持久化管理。

当前设计：
1. 以原始问答为单位追加写入
2. 不做 Python 侧 scope 绑定
3. 不做 Python 侧 stale 判断
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    InsightConfidence,
    InsightEntry,
    InsightIndex,
    InsightIndexRecord,
    InsightStats,
)

logger = logging.getLogger(__name__)


class InsightCache:
    """原始问答 Insight Cache 管理器。"""

    def __init__(self, project_root: Path, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries 必须 >= 1，当前值: {max_entries}")
        self.project_root = project_root.resolve()
        self.max_entries = max_entries
        self._insights_dir = self.project_root / ".pce" / "insights"
        self._entries_dir = self._insights_dir / "entries"
        self._index_path = self._insights_dir / "index.json"
        self._lock = asyncio.Lock()

    async def ensure_layout(self) -> None:
        def _init_fs() -> bool:
            self._insights_dir.mkdir(parents=True, exist_ok=True)
            self._entries_dir.mkdir(parents=True, exist_ok=True)
            return self._index_path.exists()

        index_exists = await asyncio.to_thread(_init_fs)
        if not index_exists:
            async with self._lock:
                exists = await asyncio.to_thread(self._index_path.exists)
                if not exists:
                    await self._save_index_unlocked(self._empty_index())

    async def upsert(
        self,
        question: str,
        answer: str,
        confidence: InsightConfidence = InsightConfidence.MEDIUM,
    ) -> str:
        """追加写入一条原始问答 Insight。"""
        question = question.strip()
        answer = answer.strip()
        if not question:
            raise ValueError("question 不能为空")
        if not answer:
            raise ValueError("answer 不能为空")

        await self.ensure_layout()
        now = datetime.now(timezone.utc)
        new_id = str(uuid.uuid4())

        async with self._lock:
            index = await self._load_index_unlocked()
            entry = InsightEntry(
                id=new_id,
                question=question,
                answer=answer,
                confidence=confidence,
                created_at=now,
            )
            await self._write_entry_unlocked(entry)
            index.records[new_id] = InsightIndexRecord(
                id=new_id,
                confidence=confidence,
                created_at=now,
            )
            index = await self._prune_lru_unlocked(index)
            await self._save_index_unlocked(index)

        logger.debug("Insight upsert 完成: id=%s", new_id)
        return new_id

    async def get_top_k(self, k: int, token_budget: int) -> tuple[str, list[str]]:
        if k <= 0 or token_budget <= 0:
            return "", []

        index = await self._load_index_safe()
        if not index.records:
            return "", []

        candidates = sorted(
            index.records.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )

        selected_ids: list[str] = []
        selected_blocks: list[str] = []
        used_chars = 0
        for record in candidates:
            if len(selected_ids) >= k:
                break
            entry = await self._read_entry(record.id)
            if entry is None:
                continue
            block = (
                f"### {entry.id}\n"
                f"- created_at: {entry.created_at.isoformat()}\n"
                f"- confidence: {entry.confidence}\n"
                f"- question: {entry.question}\n"
                f"- answer: {entry.answer}"
            )
            if used_chars + len(block) > token_budget:
                continue
            selected_ids.append(record.id)
            selected_blocks.append(block)
            used_chars += len(block)

        if not selected_ids:
            return "", []

        injected = "## 动态认知 (Insight Cache)\n\n" + "\n\n".join(selected_blocks)
        return injected, selected_ids

    async def sweep_stale(self, scopes: list[str] | None = None) -> int:
        """兼容保留：原始问答 insight 不做 stale 判断。"""
        del scopes
        await self.ensure_layout()
        return 0

    async def cleanup_stale(self) -> int:
        """轻量清理：删除不在 index 中的孤儿 entry 文件。"""
        await self.ensure_layout()
        removed = 0
        async with self._lock:
            index = await self._load_index_unlocked()
            known_ids = set(index.records.keys())
            try:
                entry_files = await asyncio.to_thread(lambda: list(self._entries_dir.glob("*.json")))
            except Exception:
                entry_files = []
            for path in entry_files:
                if path.stem not in known_ids:
                    await asyncio.to_thread(path.unlink, True)
                    removed += 1
        return removed

    async def stats(self) -> InsightStats:
        index = await self._load_index_safe()
        total = len(index.records)
        return InsightStats(
            total_entries=total,
            active_entries=total,
            stale_entries=0,
            last_updated=index.updated_at if total > 0 else None,
        )

    async def get_all_records(
        self, *, include_stale: bool = True
    ) -> list[InsightIndexRecord]:
        del include_stale
        index = await self._load_index_safe()
        records = list(index.records.values())
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    async def get_entry(self, entry_id: str) -> InsightEntry | None:
        return await self._read_entry(entry_id)

    async def delete_by_ids(self, ids: list[str]) -> int:
        normalized = [eid.strip() for eid in ids if eid and eid.strip()]
        if not normalized:
            return 0

        await self.ensure_layout()
        removed = 0
        async with self._lock:
            index = await self._load_index_unlocked()
            changed = False
            for entry_id in dict.fromkeys(normalized):
                if index.records.pop(entry_id, None) is not None:
                    removed += 1
                    changed = True
                await asyncio.to_thread(self._entry_path(entry_id).unlink, True)
            if changed:
                await self._save_index_unlocked(index)
        return removed

    def _entry_path(self, entry_id: str) -> Path:
        return self._entries_dir / f"{entry_id}.json"

    def _empty_index(self) -> InsightIndex:
        return InsightIndex(
            version="1",
            updated_at=datetime.now(timezone.utc),
            records={},
        )

    async def _load_index_safe(self) -> InsightIndex:
        if not await asyncio.to_thread(self._index_path.exists):
            return self._empty_index()
        try:
            raw = await asyncio.to_thread(self._index_path.read_text, "utf-8")
            return InsightIndex.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning(f"读取 Insight 索引失败，返回空索引: {e}")
            return self._empty_index()

    async def _load_index_unlocked(self) -> InsightIndex:
        if not await asyncio.to_thread(self._index_path.exists):
            return self._empty_index()
        try:
            raw = await asyncio.to_thread(self._index_path.read_text, "utf-8")
            return InsightIndex.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning(f"读取 Insight 索引失败，返回空索引: {e}")
            return self._empty_index()

    async def _save_index_unlocked(self, index: InsightIndex) -> None:
        index.updated_at = datetime.now(timezone.utc)
        text = json.dumps(index.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        tmp = self._insights_dir / f".index.{uuid.uuid4().hex}.tmp"
        try:
            await asyncio.to_thread(tmp.write_text, text, "utf-8")
            await asyncio.to_thread(os.replace, tmp, self._index_path)
        except Exception:
            await asyncio.to_thread(tmp.unlink, True)
            raise

    async def _write_entry_unlocked(self, entry: InsightEntry) -> None:
        path = self._entry_path(entry.id)
        text = json.dumps(entry.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            await asyncio.to_thread(tmp.write_text, text, "utf-8")
            await asyncio.to_thread(os.replace, tmp, path)
        except Exception:
            await asyncio.to_thread(tmp.unlink, True)
            raise

    async def _read_entry(self, entry_id: str) -> InsightEntry | None:
        path = self._entry_path(entry_id)
        if not await asyncio.to_thread(path.exists):
            return None
        try:
            raw = await asyncio.to_thread(path.read_text, "utf-8")
            return InsightEntry.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning(f"读取 Insight 条目失败 {entry_id}: {e}")
            return None

    async def _prune_lru_unlocked(self, index: InsightIndex) -> InsightIndex:
        overflow = len(index.records) - self.max_entries
        if overflow <= 0:
            return index

        evict = sorted(index.records.values(), key=lambda r: r.created_at)[:overflow]
        for record in evict:
            index.records.pop(record.id, None)
            await asyncio.to_thread(self._entry_path(record.id).unlink, True)
        logger.debug("LRU 淘汰 %d 条 Insight 条目", len(evict))
        return index
