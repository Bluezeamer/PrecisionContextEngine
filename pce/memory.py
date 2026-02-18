"""PCE Memory 存储模块。

负责索引、记忆与会话状态的持久化读写,所有文件操作使用原子写入保证一致性。

文件布局:
.pce/
├── meta.json              # 项目元数据
├── structure.md           # 目录职责与模块清单
├── references.json        # 符号引用索引
├── annotations.md         # 语义注解(JSONL 格式)
└── sessions/              # 会话状态
    └── {session_id}.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles

from .models import IndexSnapshot, MemoryItem, ProjectMeta, SessionState

logger = logging.getLogger(__name__)

# 常量定义
PCE_DIR_NAME = ".pce"
META_FILE = "meta.json"
STRUCTURE_FILE = "structure.md"
REFERENCES_FILE = "references.json"
ANNOTATIONS_FILE = "annotations.md"
SESSIONS_DIR = "sessions"

# Markdown 模板
STRUCTURE_TEMPLATE = """# PCE 项目结构索引

> 本文件记录项目目录职责与模块清单,便于快速理解代码组织结构。

## 目录职责

(待索引器填充)

## 顶层模块清单

(待索引器填充)
"""

ANNOTATIONS_TEMPLATE = """# PCE 语义注解与记忆

> 本文件采用 JSONL (JSON Lines) 格式存储记忆条目。

```jsonl
```
"""


# ============================================================================
# 路径辅助函数
# ============================================================================


def _resolve_root(root_path: Path | None) -> Path:
    """解析项目根路径,默认为当前工作目录。"""
    return root_path or Path.cwd()


def _pce_dir(root_path: Path) -> Path:
    """返回 .pce 目录路径。"""
    return root_path / PCE_DIR_NAME


def _meta_path(root_path: Path) -> Path:
    """返回 meta.json 文件路径。"""
    return _pce_dir(root_path) / META_FILE


def _references_path(root_path: Path) -> Path:
    """返回 references.json 文件路径。"""
    return _pce_dir(root_path) / REFERENCES_FILE


def _structure_path(root_path: Path) -> Path:
    """返回 structure.md 文件路径。"""
    return _pce_dir(root_path) / STRUCTURE_FILE


def _annotations_path(root_path: Path) -> Path:
    """返回 annotations.md 文件路径。"""
    return _pce_dir(root_path) / ANNOTATIONS_FILE


def _sessions_dir(root_path: Path) -> Path:
    """返回 sessions 目录路径。"""
    return _pce_dir(root_path) / SESSIONS_DIR


def _session_path(root_path: Path, session_id: str) -> Path:
    """返回指定会话的文件路径。"""
    return _sessions_dir(root_path) / f"{session_id}.json"


# ============================================================================
# 文件 I/O 辅助函数
# ============================================================================


async def _atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本文件(先写临时文件再重命名)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        # 使用 os.replace 保证原子性(即使目标文件存在也会覆盖)
        await asyncio.to_thread(os.replace, tmp_path, path)
        logger.debug(f"成功写入文件: {path}")
    except Exception:
        logger.exception(f"写入文件失败: {path}")
        # 清理临时文件
        if tmp_path.exists():
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
        raise


async def _atomic_write_json(path: Path, payload: Any) -> None:
    """原子写入 JSON 文件。"""
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    await _atomic_write_text(path, data + "\n")


async def _read_text(path: Path) -> str:
    """读取文本文件。"""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def _read_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    raw = await _read_text(path)
    return json.loads(raw)


# ============================================================================
# JSONL 格式处理
# ============================================================================


def _parse_jsonl_block(text: str) -> list[dict[str, Any]]:
    """从 Markdown 代码块中解析 JSONL 内容。"""
    lines = text.splitlines()
    in_block = False
    records: list[dict[str, Any]] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_block and "jsonl" in stripped:
                in_block = True
            elif in_block:
                in_block = False
            continue

        if in_block and stripped:
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效的 JSONL 行: {stripped[:50]}... 错误: {e}")
                continue

    return records


def _build_jsonl_block(existing_text: str, new_lines: list[str]) -> str:
    """在 Markdown 代码块中追加 JSONL 行。"""
    lines = existing_text.splitlines()
    start_idx = None
    end_idx = None

    # 查找 ```jsonl 代码块的位置
    for idx, line in enumerate(lines):
        if "```jsonl" in line:
            start_idx = idx
        elif start_idx is not None and line.strip().startswith("```"):
            end_idx = idx
            break

    # 如果找不到完整的代码块,尝试修复而非覆盖
    if start_idx is None:
        # 没有找到开始标记,在末尾添加完整代码块
        logger.warning("未找到 JSONL 代码块,在文件末尾创建新代码块")
        result = lines + ["", "```jsonl"] + new_lines + ["```"]
        return "\n".join(result) + "\n"

    if end_idx is None:
        # 找到开始标记但没有结束标记,补全结束标记
        logger.warning("JSONL 代码块缺少结束标记,自动补全")
        end_idx = len(lines)
        result = lines[:end_idx] + new_lines + ["```"]
        return "\n".join(result) + "\n"

    # 正常情况:在代码块结束标记前插入新行
    result = lines[:end_idx] + new_lines + lines[end_idx:]
    return "\n".join(result) + "\n"


# ============================================================================
# 初始化布局
# ============================================================================


async def _ensure_layout(root_path: Path) -> None:
    """确保 .pce 目录结构存在,初始化必要的模板文件。"""
    pce_dir = _pce_dir(root_path)
    pce_dir.mkdir(parents=True, exist_ok=True)
    _sessions_dir(root_path).mkdir(parents=True, exist_ok=True)

    # 初始化 structure.md
    structure_path = _structure_path(root_path)
    if not structure_path.exists():
        await _atomic_write_text(structure_path, STRUCTURE_TEMPLATE)
        logger.debug(f"初始化结构文件: {structure_path}")

    # 初始化 annotations.md
    annotations_path = _annotations_path(root_path)
    if not annotations_path.exists():
        await _atomic_write_text(annotations_path, ANNOTATIONS_TEMPLATE)
        logger.debug(f"初始化注解文件: {annotations_path}")


# ============================================================================
# 索引管理
# ============================================================================


async def load_index(root_path: Path | None = None) -> IndexSnapshot | None:
    """加载项目索引快照。

    Args:
        root_path: 项目根路径,默认为当前目录

    Returns:
        索引快照对象,如果不存在则返回 None

    Raises:
        Exception: 文件读取或解析失败
    """
    root_path = _resolve_root(root_path)
    path = _references_path(root_path)

    if not path.exists():
        logger.debug(f"索引文件不存在: {path}")
        return None

    try:
        data = await _read_json(path)
        snapshot = IndexSnapshot.model_validate(data)
        logger.info(f"成功加载索引,包含 {len(snapshot.entries)} 个文件")
        return snapshot
    except Exception:
        logger.exception(f"读取索引失败: {path}")
        raise


async def save_index(snapshot: IndexSnapshot, root_path: Path | None = None) -> None:
    """保存项目索引快照。

    Args:
        snapshot: 要保存的索引快照
        root_path: 项目根路径,默认为当前目录

    Raises:
        Exception: 文件写入失败
    """
    root_path = _resolve_root(root_path)
    await _ensure_layout(root_path)

    references_path = _references_path(root_path)
    meta_path = _meta_path(root_path)

    try:
        # 顺序写入以保证一致性(先写完整快照,再写元数据)
        # 如果第二步失败,第一步已写入的数据仍可用
        await _atomic_write_json(references_path, snapshot.model_dump(mode="json"))
        await _atomic_write_json(meta_path, snapshot.project_meta.model_dump(mode="json"))
        logger.info(f"成功保存索引: {references_path}")
    except Exception:
        logger.exception(f"保存索引失败: {references_path}")
        raise


async def index_exists(root_path: Path | None = None) -> bool:
    """检查索引是否已存在。

    Args:
        root_path: 项目根路径,默认为当前目录

    Returns:
        索引是否存在
    """
    root_path = _resolve_root(root_path)
    return _references_path(root_path).exists()


# ============================================================================
# Memory 管理
# ============================================================================


def _apply_memory_filters(
    items: list[MemoryItem], filters: dict[str, Any] | None
) -> list[MemoryItem]:
    """根据过滤条件筛选记忆条目。"""
    if not filters:
        return items

    result: list[MemoryItem] = []
    item_type = filters.get("item_type")
    tags = set(filters.get("tags", []))
    related_files = {str(Path(p)) for p in filters.get("related_files", [])}
    keyword = filters.get("keyword")

    for item in items:
        # 类型过滤
        if item_type and item.item_type.value != item_type:
            continue

        # 标签过滤(要求所有指定标签都存在)
        if tags and not tags.issubset(set(item.tags)):
            continue

        # 文件过滤(要求至少有一个文件匹配)
        if related_files:
            item_files = {str(path) for path in item.related_files}
            if item_files.isdisjoint(related_files):
                continue

        # 关键词过滤
        if keyword and keyword not in item.content:
            continue

        result.append(item)

    return result


async def list_memory_items(
    filters: dict[str, Any] | None = None, root_path: Path | None = None
) -> list[MemoryItem]:
    """列出所有记忆条目,支持过滤。

    Args:
        filters: 过滤条件字典,支持的 key:
            - item_type: 记忆类型(str)
            - tags: 标签列表(list[str])
            - related_files: 关联文件列表(list[str])
            - keyword: 内容关键词(str)
        root_path: 项目根路径,默认为当前目录

    Returns:
        符合条件的记忆条目列表
    """
    root_path = _resolve_root(root_path)
    path = _annotations_path(root_path)

    if not path.exists():
        logger.debug(f"注解文件不存在: {path}")
        return []

    try:
        content = await _read_text(path)
        raw_items = _parse_jsonl_block(content)
        items = [MemoryItem.model_validate(item) for item in raw_items]
        filtered = _apply_memory_filters(items, filters)
        logger.debug(f"加载 {len(items)} 条记忆,过滤后 {len(filtered)} 条")
        return filtered
    except Exception:
        logger.exception(f"读取记忆失败: {path}")
        raise


async def append_memory(item: MemoryItem, root_path: Path | None = None) -> None:
    """追加一条记忆条目。

    Args:
        item: 要追加的记忆条目
        root_path: 项目根路径,默认为当前目录

    Raises:
        Exception: 文件写入失败
    """
    root_path = _resolve_root(root_path)
    await _ensure_layout(root_path)
    path = _annotations_path(root_path)

    try:
        try:
            text = await _read_text(path)
        except FileNotFoundError:
            text = ANNOTATIONS_TEMPLATE

        payload = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        updated = _build_jsonl_block(text, [payload])
        await _atomic_write_text(path, updated)
        logger.debug(f"成功追加记忆: {item.item_id}")
    except Exception:
        logger.exception(f"追加记忆失败: {path}")
        raise


async def clear_memory(root_path: Path | None = None) -> None:
    """清空所有记忆条目。

    Args:
        root_path: 项目根路径,默认为当前目录

    Raises:
        Exception: 文件写入失败
    """
    root_path = _resolve_root(root_path)
    await _ensure_layout(root_path)
    path = _annotations_path(root_path)

    try:
        await _atomic_write_text(path, ANNOTATIONS_TEMPLATE)
        logger.info(f"成功清空记忆: {path}")
    except Exception:
        logger.exception(f"清空记忆失败: {path}")
        raise


# ============================================================================
# 会话管理
# ============================================================================


async def load_session(session_id: str, root_path: Path | None = None) -> SessionState | None:
    """加载会话状态。

    Args:
        session_id: 会话 ID
        root_path: 项目根路径,默认为当前目录

    Returns:
        会话状态对象,如果不存在则返回 None

    Raises:
        Exception: 文件读取或解析失败
    """
    root_path = _resolve_root(root_path)
    path = _session_path(root_path, session_id)

    if not path.exists():
        logger.debug(f"会话文件不存在: {path}")
        return None

    try:
        data = await _read_json(path)
        state = SessionState.model_validate(data)
        logger.debug(f"成功加载会话: {session_id}")
        return state
    except Exception:
        logger.exception(f"读取会话失败: {path}")
        raise


async def save_session(state: SessionState, root_path: Path | None = None) -> None:
    """保存会话状态。

    Args:
        state: 要保存的会话状态
        root_path: 项目根路径,默认为当前目录

    Raises:
        Exception: 文件写入失败
    """
    root_path = _resolve_root(root_path)
    await _ensure_layout(root_path)
    path = _session_path(root_path, state.session_id)

    try:
        await _atomic_write_json(path, state.model_dump(mode="json"))
        logger.debug(f"成功保存会话: {state.session_id}")
    except Exception:
        logger.exception(f"保存会话失败: {path}")
        raise


async def clear_session(session_id: str, root_path: Path | None = None) -> None:
    """删除会话状态文件。

    Args:
        session_id: 会话 ID
        root_path: 项目根路径,默认为当前目录

    Raises:
        Exception: 文件删除失败
    """
    root_path = _resolve_root(root_path)
    path = _session_path(root_path, session_id)

    if not path.exists():
        logger.debug(f"会话文件不存在,无需删除: {path}")
        return

    try:
        await asyncio.to_thread(path.unlink)
        logger.debug(f"成功删除会话: {session_id}")
    except Exception:
        logger.exception(f"删除会话失败: {path}")
        raise


# ============================================================================
# 状态查询
# ============================================================================


async def get_status(root_path: Path | None = None) -> dict[str, Any]:
    """获取 PCE 当前状态。

    Args:
        root_path: 项目根路径,默认为当前目录

    Returns:
        状态字典,包含:
            - last_index_time: 最后索引时间
            - index_version: 索引版本号
            - memory_items_count: 记忆条目数量
    """
    root_path = _resolve_root(root_path)
    meta_path = _meta_path(root_path)

    last_index_time = None
    index_version = None

    if meta_path.exists():
        try:
            meta = ProjectMeta.model_validate(await _read_json(meta_path))
            last_index_time = meta.created_at
            index_version = meta.index_version
        except Exception:
            logger.exception(f"读取项目元数据失败: {meta_path}")
            # 不抛出异常,返回部分状态信息

    memory_items = await list_memory_items(root_path=root_path)

    return {
        "last_index_time": last_index_time.isoformat() if last_index_time else None,
        "index_version": index_version,
        "memory_items_count": len(memory_items),
    }
