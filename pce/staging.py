"""文件变更暂存区管理。

监听项目目录中的代码文件变更，将脏文件记录到 .pce/dirty_files.json。
PCE Agent 可以在后续查询中感知这些变更，并通过认知确认机制更新状态。

暂存区条目生命周期：
    文件被修改 → record_change(ack=False)
    PCE Agent 探索并理解变更 → acknowledge(ack=True)
    同一文件再次被修改 → record_change 覆盖 last_hash，需重新认知
    新会话启动 → list_unacknowledged() 获取待处理文件
    增量索引完成 → acknowledge_after_reindex() 批量确认

存储路径: {project_root}/.pce/dirty_files.json
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from watchfiles import Change, awatch

from .file_discovery import (
    HARD_SKIP_DIRS,
    should_track_deleted_path,
    should_track_existing_file,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================


@dataclass(frozen=True)
class DirtyState:
    """暂存区查询结果：待处理的变更和删除文件列表。"""

    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.changed and not self.deleted


# ============================================================================
# 暂存区管理
# ============================================================================


class StagingArea:
    """文件变更暂存区，持久化到 .pce/dirty_files.json。

    所有写操作通过 asyncio.Lock 保证原子性。
    JSON schema::

        {
          "version": 1,
          "files": {
            "pce/server.py": {
              "last_hash": "sha256hex",
              "acknowledged_hash": "sha256hex" | null,
              "deleted": false,
              "detected_at": "ISO8601",
              "acknowledged_at": "ISO8601" | null
            }
          }
        }
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._path = self.project_root / ".pce" / "dirty_files.json"
        self._lock = asyncio.Lock()

    # ------ 公开 API ------

    async def record_change(self, rel_path: str, *, deleted: bool = False) -> None:
        """记录一个文件变更（或删除）到暂存区。

        条目语义：存在于暂存区 = Memory 尚未包含该文件的最新认知。
        增量索引完成后由 acknowledge_after_reindex 移除。
        """
        if deleted:
            if not should_track_deleted_path(rel_path):
                return
        else:
            if not should_track_existing_file(self.project_root, rel_path):
                return

        async with self._lock:
            payload = await self._read()
            files: dict[str, Any] = payload.setdefault("files", {})
            now = datetime.now(UTC).isoformat()

            if deleted:
                files[rel_path] = {
                    "last_hash": None,
                    "deleted": True,
                    "detected_at": now,
                }
            else:
                abs_path = self.project_root / rel_path
                file_hash = await _hash_file(abs_path) if abs_path.exists() else None
                files[rel_path] = {
                    "last_hash": file_hash,
                    "deleted": False,
                    "detected_at": now,
                }

            await self._write(payload)

    async def list_unacknowledged(self) -> DirtyState:
        """列出会话内尚未被 PCE Agent 认知的变更文件。

        排除 session_acknowledged=True 的条目（Agent 本次会话已读取理解过），
        避免同一对话内重复提示。用于注入脏文件上下文到 query。
        """
        async with self._lock:
            payload = await self._read()

        files = payload.get("files", {})
        changed: list[str] = []
        deleted: list[str] = []

        for path, record in files.items():
            # 跳过本次会话已认知的条目
            if record.get("session_acknowledged", False):
                continue

            if record.get("deleted", False):
                deleted.append(path)
            else:
                changed.append(path)

        return DirtyState(changed=changed, deleted=deleted)

    async def list_pending_reindex(self) -> DirtyState:
        """列出所有待增量索引的文件（暂存区中的全部条目）。

        与 list_unacknowledged 不同，此方法不关心 session_acknowledged 标记。
        只要条目仍在暂存区中，就说明 Memory 尚未包含其最新认知，需要索引。
        供 _ensure_initialized 在会话开始时使用。
        """
        async with self._lock:
            payload = await self._read()

        files = payload.get("files", {})
        changed: list[str] = []
        deleted: list[str] = []

        for path, record in files.items():
            if record.get("deleted", False):
                deleted.append(path)
            else:
                changed.append(path)

        return DirtyState(changed=changed, deleted=deleted)

    async def snapshot_hashes(self, paths: list[str]) -> dict[str, str | None]:
        """获取指定文件当前的 last_hash 快照，用于增量索引后的一致性校验。"""
        async with self._lock:
            payload = await self._read()
        files = payload.get("files", {})
        return {p: files.get(p, {}).get("last_hash") for p in paths}

    async def acknowledge(self, paths: list[str]) -> None:
        """PCE Agent 会话内标记已认知指定文件的变更。

        这是临时认知标记：Agent 在当前上下文中已读取并理解了变更，
        同一会话内不再重复提示。但条目保留在暂存区，
        留待下次会话通过增量索引沉淀到 Memory 后才移除。

        注意：不修改 acknowledged_hash，以免影响 list_pending_reindex 的判断。
        """
        if not paths:
            return
        async with self._lock:
            payload = await self._read()
            files = payload.get("files", {})
            now = datetime.now(UTC).isoformat()

            for path in paths:
                record = files.get(path)
                if not record:
                    continue
                record["session_acknowledged"] = True
                record["session_acknowledged_at"] = now

            await self._write(payload)

    async def acknowledge_after_reindex(
        self,
        paths: list[str],
        expected_hashes: dict[str, str | None] | None = None,
    ) -> None:
        """增量索引完成后批量确认并清理暂存区。

        索引更新完成后，Memory 已包含最新认知，对应的暂存区条目使命结束，
        直接从 JSON 中移除。新对话无需知道历史变更轨迹。

        Args:
            paths: 需要确认的文件路径列表
            expected_hashes: 索引开始前的 hash 快照。仅当 last_hash 仍匹配时才确认，
                           防止索引期间新变更被"提前确认"。为 None 时无条件确认。
        """
        if not paths:
            return
        async with self._lock:
            payload = await self._read()
            files = payload.get("files", {})

            for path in paths:
                record = files.get(path)
                if not record:
                    continue

                # hash 一致性校验：索引期间文件再次变更则跳过确认
                if expected_hashes is not None:
                    expected = expected_hashes.get(path)
                    current = record.get("last_hash")
                    if current != expected:
                        logger.info(
                            f"跳过确认 {path}: 索引期间文件再次变更 "
                            f"(expected={expected!r:.16}, current={current!r:.16})"
                        )
                        continue

                # 索引已更新，条目使命完成，从暂存区移除
                del files[path]

            await self._write(payload)

    async def summary(self) -> dict[str, Any]:
        """返回暂存区概要统计（用于 pce_status）。"""
        async with self._lock:
            payload = await self._read()
        files = payload.get("files", {})
        # 单次遍历：暂存区中所有条目都是待索引的
        n_changed = 0
        n_deleted = 0
        n_session_acked = 0
        for record in files.values():
            if record.get("deleted", False):
                n_deleted += 1
            else:
                n_changed += 1
            if record.get("session_acknowledged", False):
                n_session_acked += 1
        return {
            "pending_reindex": len(files),
            "pending_changed": n_changed,
            "pending_deleted": n_deleted,
            "session_acknowledged": n_session_acked,
        }

    # ------ 内部方法 ------

    async def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"version": 1, "files": {}}
        text = await asyncio.to_thread(self._path.read_text, "utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"暂存区文件解析失败，重置: {self._path}")
            return {"version": 1, "files": {}}

    async def _write(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        # 原子写入：先写临时文件，再 rename 替换，防止中断导致 JSON 损坏
        tmp_path = self._path.with_suffix(".tmp")
        await asyncio.to_thread(tmp_path.write_text, text, "utf-8")
        await asyncio.to_thread(os.replace, tmp_path, self._path)


# ============================================================================
# 文件监听
# ============================================================================


class FileWatcher:
    """基于 watchfiles 的文件变更监听器，将变更写入 StagingArea。"""

    def __init__(self, staging: StagingArea) -> None:
        self._staging = staging
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """启动后台文件监听任务。"""
        if self.running:
            return
        self._task = asyncio.create_task(self._watch_loop(), name="pce-file-watcher")
        logger.info(f"文件监听已启动: {self._staging.project_root}")

    async def stop(self) -> None:
        """停止后台文件监听任务。"""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("文件监听已停止")

    async def _watch_loop(self) -> None:
        """监听文件变更事件并记录到暂存区。"""
        root = self._staging.project_root
        try:
            async for changes in awatch(
                root,
                watch_filter=_watch_filter,
                # debounce 300ms，聚合快速连续变更
                debounce=300,
                # 不递归进入 SKIP_DIRS
                ignore_permission_denied=True,
            ):
                for change_type, path_str in changes:
                    path = Path(path_str)
                    try:
                        rel = path.relative_to(root)
                    except ValueError:
                        continue

                    rel_str = str(rel)
                    if change_type == Change.deleted:
                        await self._staging.record_change(rel_str, deleted=True)
                        logger.debug(f"文件删除: {rel_str}")
                    else:
                        # Change.added 或 Change.modified 都视为变更
                        await self._staging.record_change(rel_str)
                        logger.debug(f"文件变更: {rel_str}")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("文件监听异常")


# ============================================================================
# 辅助函数
# ============================================================================


def _watch_filter(change: Change, path: str) -> bool:
    """watchfiles 过滤器：只过滤内建硬规则目录，其余交给 record_change 判定。"""
    p = Path(path)
    return not any(part in HARD_SKIP_DIRS for part in p.parts)


async def _hash_file(path: Path) -> str:
    """计算文件的 SHA-256 hash（通过 to_thread 避免阻塞事件循环）。"""

    def _compute() -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    return await asyncio.to_thread(_compute)
