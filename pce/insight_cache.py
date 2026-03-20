"""Insight Cache 持久化管理。

负责管理 `.pce/insights/` 下的动态认知缓存，包括：
1. 条目写入与覆盖更新（upsert）
2. 查询时的注入文本构建（get_top_k）
3. stale 检测、批量清理与统计

存储结构：
    {project_root}/.pce/insights/
    ├── index.json       # InsightIndex 序列化
    └── entries/
        └── {uuid}.json  # InsightEntry 序列化

并发安全：
    所有写操作持 asyncio.Lock。
    读操作无锁，依赖 os.replace 的原子性。
    _unlocked 后缀方法为持锁后调用的内部版本。
"""

from __future__ import annotations

import asyncio
import hashlib
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
    """Insight Cache 管理器。

    懒加载：__init__ 不做 I/O，所有 I/O 在首次使用时通过 ensure_layout 初始化。
    """

    def __init__(self, project_root: Path, max_entries: int = 100) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries 必须 >= 1，当前值: {max_entries}")
        self.project_root = project_root.resolve()
        self.max_entries = max_entries
        self._insights_dir = self.project_root / ".pce" / "insights"
        self._entries_dir = self._insights_dir / "entries"
        self._index_path = self._insights_dir / "index.json"
        self._lock = asyncio.Lock()

    # =========================================================================
    # 公开接口
    # =========================================================================

    async def ensure_layout(self) -> None:
        """确保目录结构存在，必要时写入空索引。幂等。"""
        def _init_fs() -> bool:
            self._insights_dir.mkdir(parents=True, exist_ok=True)
            self._entries_dir.mkdir(parents=True, exist_ok=True)
            return self._index_path.exists()

        index_exists = await asyncio.to_thread(_init_fs)
        if not index_exists:
            async with self._lock:
                # 双重检查：持锁后确认仍不存在
                exists = await asyncio.to_thread(self._index_path.exists)
                if not exists:
                    await self._save_index_unlocked(self._empty_index())

    async def upsert(
        self,
        scope: str,
        content: str,
        confidence: InsightConfidence = InsightConfidence.MEDIUM,
        source_hash: str | None = None,
    ) -> str:
        """写入或覆盖一个 scope 的 Insight 条目。

        若 source_hash 未提供，自动读取 scope 对应文件计算。
        同 scope 已有条目时，旧条目被逻辑删除（entry 文件同步删除）。
        返回新条目 ID。
        """
        scope = scope.strip()
        content = content.strip()
        if not scope:
            raise ValueError("scope 不能为空")
        if not content:
            raise ValueError("content 不能为空")

        await self.ensure_layout()

        if source_hash is None:
            source_hash = await asyncio.to_thread(self._compute_hash, scope)
        if source_hash is None:
            raise FileNotFoundError(f"scope 对应文件不存在: {scope}")

        now = datetime.now(timezone.utc)
        new_id = str(uuid.uuid4())

        async with self._lock:
            index = await self._load_index_unlocked()

            # 查找并清除同 scope 的旧活跃条目
            previous_id: str | None = None
            for record in list(index.records.values()):
                if record.scope == scope:
                    previous_id = record.id
                    index.records.pop(previous_id)
                    await asyncio.to_thread(self._entry_path(previous_id).unlink, True)
                    break  # scope 唯一，找到即止

            # 写入新 entry 文件
            entry = InsightEntry(
                id=new_id,
                scope=scope,
                content=content,
                confidence=confidence,
                created_at=now,
                last_referenced_at=now,
                source_hash=source_hash,
                supersedes=previous_id,
            )
            await self._write_entry_unlocked(entry)

            # 更新索引
            index.records[new_id] = InsightIndexRecord(
                id=new_id,
                scope=scope,
                confidence=confidence,
                created_at=now,
                last_referenced_at=now,
                source_hash=source_hash,
                supersedes=previous_id,
                stale=False,
                stale_checked_at=now,
            )

            index = await self._prune_lru_unlocked(index)
            await self._save_index_unlocked(index)

        logger.debug(f"Insight upsert 完成: scope={scope}, id={new_id}")
        return new_id

    async def get_top_k(self, k: int, token_budget: int) -> tuple[str, list[str]]:
        """取 top-k 条目构建注入文本。

        优先取非 stale 条目，预算不足时跳过而非截断。
        被选中的条目 last_referenced_at 会被更新（touch）。

        Returns:
            (注入文本块, 被选中的条目 id 列表)
            注入文本为空时返回 ("", [])。
        """
        if k <= 0 or token_budget <= 0:
            return "", []

        # 无锁读，失败返回空
        index = await self._load_index_safe()
        if not index.records:
            return "", []

        # 按 last_referenced_at 倒序排列所有候选
        candidates = sorted(
            index.records.values(),
            key=lambda r: r.last_referenced_at,
            reverse=True,
        )

        # 懒 stale 校验：对每个候选检查当前文件 hash
        stale_updates: dict[str, bool] = {}  # id -> 新 stale 状态
        for record in candidates:
            current_hash = await asyncio.to_thread(self._compute_hash, record.scope)
            is_stale = current_hash is None or current_hash != record.source_hash
            if is_stale != record.stale:
                stale_updates[record.id] = is_stale

        # 将 stale 变化回写索引（持锁）
        # 注意：无锁读快照后持锁回写存在轻微竞态（最终一致），可接受。
        if stale_updates:
            async with self._lock:
                fresh_index = await self._load_index_unlocked()
                now = datetime.now(timezone.utc)
                index_changed = False
                for entry_id, new_stale in stale_updates.items():
                    rec = fresh_index.records.get(entry_id)
                    if rec is not None:
                        rec.stale = new_stale
                        rec.stale_checked_at = now
                        index_changed = True
                if index_changed:
                    await self._save_index_unlocked(fresh_index)
                # 用最新索引重建候选列表
                candidates = sorted(
                    fresh_index.records.values(),
                    key=lambda r: r.last_referenced_at,
                    reverse=True,
                )

        # 分拣 fresh / stale
        fresh = [r for r in candidates if not r.stale]
        stale = [r for r in candidates if r.stale]

        selected_ids: list[str] = []
        selected_blocks: list[str] = []
        used_chars = 0

        async def _try_add(records: list[InsightIndexRecord]) -> None:
            nonlocal used_chars
            for record in records:
                if len(selected_ids) >= k:
                    return
                entry = await self._read_entry(record.id)
                if entry is None:
                    continue
                if used_chars + len(entry.content) > token_budget:
                    continue
                title = f"### {entry.scope}" + (" [可能过时]" if record.stale else "")
                selected_ids.append(record.id)
                selected_blocks.append(f"{title}\n{entry.content}")
                used_chars += len(entry.content)

        await _try_add(fresh)
        if len(selected_ids) < k:
            await _try_add(stale)

        if not selected_ids:
            return "", []

        # 更新被选条目的 last_referenced_at
        await self._touch_entries(selected_ids)

        injected = "## 动态认知 (Insight Cache)\n\n" + "\n\n".join(selected_blocks)
        return injected, selected_ids

    async def sweep_stale(self, scopes: list[str] | None = None) -> int:
        """批量检测并标记 stale，返回新标记为 stale 的数量。

        scopes 为 None 时检查所有活跃条目。
        """
        await self.ensure_layout()
        scope_filter = {s.strip() for s in scopes if s.strip()} if scopes else None
        new_stale_count = 0

        async with self._lock:
            index = await self._load_index_unlocked()
            now = datetime.now(timezone.utc)
            changed = False

            for record in index.records.values():
                if scope_filter is not None and record.scope not in scope_filter:
                    continue
                current_hash = await asyncio.to_thread(self._compute_hash, record.scope)
                is_stale = current_hash is None or current_hash != record.source_hash
                if is_stale and not record.stale:
                    new_stale_count += 1
                if record.stale != is_stale:
                    record.stale = is_stale
                    changed = True
                # stale_checked_at 每次都更新，需持久化
                record.stale_checked_at = now
                changed = True

            if changed:
                await self._save_index_unlocked(index)

        return new_stale_count

    async def cleanup_stale(self) -> int:
        """删除所有 stale 条目及其 entry 文件，并清理孤儿 entry 文件。

        返回总删除数量（stale 条目 + 孤儿文件）。
        """
        await self.ensure_layout()
        removed = 0

        async with self._lock:
            index = await self._load_index_unlocked()

            # 1. 删除 index 中标记为 stale 的条目
            stale_ids = [rid for rid, rec in index.records.items() if rec.stale]
            for entry_id in stale_ids:
                index.records.pop(entry_id)
                await asyncio.to_thread(self._entry_path(entry_id).unlink, True)
            removed += len(stale_ids)

            # 2. 扫描 entries 目录，删除不在 index 中的孤儿文件
            known_ids = set(index.records.keys())
            try:
                entry_files = await asyncio.to_thread(
                    lambda: list(self._entries_dir.glob("*.json"))
                )
            except Exception:
                entry_files = []
            for path in entry_files:
                file_id = path.stem
                if file_id not in known_ids:
                    await asyncio.to_thread(path.unlink, True)
                    removed += 1

            if stale_ids:
                await self._save_index_unlocked(index)

        if removed:
            logger.debug("cleanup_stale 删除 %d 条（stale %d + 孤儿 %d）",
                         removed, len(stale_ids), removed - len(stale_ids))
        return removed

    async def stats(self) -> InsightStats:
        """返回 Insight Cache 统计信息。

        active_entries = 非 stale 条目数（可直接注入 prompt 的条目）
        stale_entries  = 已过时但尚未清理的条目数
        """
        index = await self._load_index_safe()
        total = len(index.records)
        stale = sum(1 for r in index.records.values() if r.stale)
        return InsightStats(
            total_entries=total,
            active_entries=total - stale,
            stale_entries=stale,
            last_updated=index.updated_at if total > 0 else None,
        )

    async def get_all_records(
        self, *, include_stale: bool = True
    ) -> list[InsightIndexRecord]:
        """返回全部索引记录，默认包含 stale 条目。DigestAgent 用于构建任务清单。"""
        index = await self._load_index_safe()
        records = list(index.records.values())
        if not include_stale:
            records = [r for r in records if not r.stale]
        return sorted(records, key=lambda r: r.last_referenced_at, reverse=True)

    async def get_entry_content(self, entry_id: str) -> str | None:
        """返回指定条目的正文内容，条目不存在时返回 None。"""
        entry = await self._read_entry(entry_id)
        return entry.content if entry is not None else None

    async def delete_by_ids(self, ids: list[str]) -> int:
        """按 ID 批量删除条目及其 entry 文件，返回实际从索引中删除的数量。"""
        normalized = [eid.strip() for eid in ids if eid and eid.strip()]
        if not normalized:
            return 0

        await self.ensure_layout()
        removed = 0
        async with self._lock:
            index = await self._load_index_unlocked()
            changed = False
            for entry_id in dict.fromkeys(normalized):  # 去重保序
                if index.records.pop(entry_id, None) is not None:
                    removed += 1
                    changed = True
                await asyncio.to_thread(self._entry_path(entry_id).unlink, True)
            if changed:
                await self._save_index_unlocked(index)
        return removed

    # =========================================================================
    # 私有辅助
    # =========================================================================

    def _entry_path(self, entry_id: str) -> Path:
        return self._entries_dir / f"{entry_id}.json"

    def _compute_hash(self, scope: str) -> str | None:
        """计算 scope 对应文件的 SHA256，文件不存在时返回 None。同步，供 to_thread 调用。"""
        path = self.project_root / scope
        if not path.is_file():
            return None
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _empty_index(self) -> InsightIndex:
        return InsightIndex(
            version="1",
            updated_at=datetime.now(timezone.utc),
            records={},
        )

    async def _load_index_safe(self) -> InsightIndex:
        """无锁读索引，任何失败均返回空索引（不持锁，适合读路径）。"""
        if not await asyncio.to_thread(self._index_path.exists):
            return self._empty_index()
        try:
            raw = await asyncio.to_thread(self._index_path.read_text, "utf-8")
            return InsightIndex.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning(f"读取 Insight 索引失败，返回空索引: {e}")
            return self._empty_index()

    async def _load_index_unlocked(self) -> InsightIndex:
        """读索引（调用方需持锁）。失败时返回空索引。"""
        if not await asyncio.to_thread(self._index_path.exists):
            return self._empty_index()
        try:
            raw = await asyncio.to_thread(self._index_path.read_text, "utf-8")
            return InsightIndex.model_validate(json.loads(raw))
        except Exception as e:
            logger.warning(f"读取 Insight 索引失败，返回空索引: {e}")
            return self._empty_index()

    async def _save_index_unlocked(self, index: InsightIndex) -> None:
        """原子写索引（调用方需持锁）。使用临时文件 + os.replace 保证原子性。"""
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
        """写入 entry 文件（调用方需持锁）。原子写。"""
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
        """读取单条 InsightEntry，失败返回 None。无锁。"""
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
        """超出容量时 LRU 淘汰最旧条目（调用方需持锁）。"""
        overflow = len(index.records) - self.max_entries
        if overflow <= 0:
            return index

        evict = sorted(index.records.values(), key=lambda r: r.last_referenced_at)[:overflow]
        for record in evict:
            index.records.pop(record.id)
            await asyncio.to_thread(self._entry_path(record.id).unlink, True)

        logger.debug(f"LRU 淘汰 {len(evict)} 条 Insight 条目")
        return index

    async def _touch_entries(self, ids: list[str]) -> None:
        """更新指定条目的 last_referenced_at（持锁）。"""
        if not ids:
            return
        id_set = set(ids)
        async with self._lock:
            index = await self._load_index_unlocked()
            now = datetime.now(timezone.utc)
            changed = False
            for record in index.records.values():
                if record.id in id_set:
                    record.last_referenced_at = now
                    changed = True
            if changed:
                await self._save_index_unlocked(index)
