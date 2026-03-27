"""轻量 baseline 维护辅助。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from .memory import load_index, save_file_baseline

logger = logging.getLogger(__name__)


async def seed_initial_file_baselines_if_missing(*, project_root: Path) -> None:
    """若当前不存在任何 baseline，则用现有索引建立初始 baseline。"""
    baselines_dir = project_root / ".pce" / "baselines" / "files"
    if baselines_dir.exists():
        existing = list(baselines_dir.rglob("*.json"))
        if existing:
            return

    snapshot = await load_index(root_path=project_root)
    if snapshot is None:
        return

    for entry in snapshot.entries:
        rel_path = str(entry.file_meta.path)
        abs_path = project_root / rel_path
        try:
            content = await asyncio.to_thread(abs_path.read_text, "utf-8")
        except Exception:
            logger.warning("初始化 baseline 失败，跳过文件: %s", rel_path)
            continue
        await save_file_baseline(
            rel_path,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            symbols=entry.symbols,
            root_path=project_root,
        )
