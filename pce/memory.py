"""PCE Memory 存储模块。

负责索引与状态元数据的持久化读写,所有文件操作使用原子写入保证一致性。

文件布局:
.pce/
├── meta.json              # 项目元数据
├── structure.md           # 目录职责与模块清单
├── references.json        # 符号引用索引
└── annotations/
    ├── index.md           # 认知导航首页
    └── modules/
        └── *.md           # 模块认知文档
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles

from .models import FileBaseline, IndexSnapshot, ModuleRegistry, ProjectMeta, SymbolFact

logger = logging.getLogger(__name__)

# 常量定义
PCE_DIR_NAME = ".pce"
META_FILE = "meta.json"
STRUCTURE_FILE = "structure.md"
REFERENCES_FILE = "references.json"
MODULE_REGISTRY_FILE = "module_registry.json"
BASELINES_DIR = "baselines"
BASELINE_FILES_DIR = "files"
ANNOTATIONS_DIR = "annotations"
ANNOTATIONS_MODULES_DIR = "modules"

# Markdown 模板
STRUCTURE_TEMPLATE = """# PCE 项目结构索引

> 本文件记录项目目录职责与模块清单,便于快速理解代码组织结构。

## 目录职责

(待索引器填充)

## 顶层模块清单

(待索引器填充)
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


def _module_registry_path(root_path: Path) -> Path:
    """返回 module_registry.json 文件路径。"""
    return _pce_dir(root_path) / MODULE_REGISTRY_FILE


def _baselines_dir(root_path: Path) -> Path:
    """返回 .pce/baselines 目录路径。"""
    return _pce_dir(root_path) / BASELINES_DIR


def _baseline_files_dir(root_path: Path) -> Path:
    """返回 .pce/baselines/files 目录路径。"""
    return _baselines_dir(root_path) / BASELINE_FILES_DIR


def _baseline_file_path(root_path: Path, rel_path: str | Path) -> Path:
    """返回指定相对路径对应的 baseline 文件路径。"""
    rel = Path(rel_path)
    return _baseline_files_dir(root_path) / rel.parent / f"{rel.name}.json"


def _structure_path(root_path: Path) -> Path:
    """返回 structure.md 文件路径。"""
    return _pce_dir(root_path) / STRUCTURE_FILE


def _annotations_dir(root_path: Path) -> Path:
    """返回 .pce/annotations 目录路径。"""
    return _pce_dir(root_path) / ANNOTATIONS_DIR


def _annotation_modules_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/modules 目录路径。"""
    return _annotations_dir(root_path) / ANNOTATIONS_MODULES_DIR


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
# 初始化布局
# ============================================================================


async def _ensure_layout(root_path: Path) -> None:
    """确保 .pce 目录结构存在,初始化必要的模板文件。"""
    pce_dir = _pce_dir(root_path)
    pce_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 structure.md
    structure_path = _structure_path(root_path)
    if not structure_path.exists():
        await _atomic_write_text(structure_path, STRUCTURE_TEMPLATE)
        logger.debug(f"初始化结构文件: {structure_path}")

    registry_path = _module_registry_path(root_path)
    if not registry_path.exists():
        await _atomic_write_json(
            registry_path,
            ModuleRegistry().model_dump(mode="json"),
        )
        logger.debug(f"初始化模块注册表: {registry_path}")

    _baseline_files_dir(root_path).mkdir(parents=True, exist_ok=True)


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
            - annotation_modules_count: 已生成的模块认知文档数量
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

    annotation_modules_count = 0
    modules_dir = _annotation_modules_dir(root_path)
    if modules_dir.exists():
        try:
            annotation_modules_count = sum(
                1 for p in modules_dir.iterdir() if p.is_file() and p.suffix == ".md"
            )
        except Exception:
            logger.exception(f"读取模块认知文档目录失败: {modules_dir}")

    module_count = 0
    registry_path = _module_registry_path(root_path)
    if registry_path.exists():
        try:
            registry = ModuleRegistry.model_validate(await _read_json(registry_path))
            module_count = len(registry.records)
        except Exception:
            logger.exception(f"读取模块注册表失败: {registry_path}")

    return {
        "last_index_time": last_index_time.isoformat() if last_index_time else None,
        "index_version": index_version,
        "annotation_modules_count": annotation_modules_count,
        "module_count": module_count,
    }


async def get_module_annotation(
    module_slug: str,
    root_path: Path | None = None,
) -> str | None:
    """读取模块认知文档内容,不存在时返回 None。"""
    root_path = _resolve_root(root_path)
    path = _annotation_modules_dir(root_path) / f"{module_slug}.md"
    if not path.exists():
        return None
    try:
        return await _read_text(path)
    except Exception:
        logger.exception(f"读取模块认知文档失败: {path}")
        return None


async def save_file_baseline(
    rel_path: str | Path,
    *,
    content: str,
    content_hash: str,
    symbols: list[SymbolFact],
    root_path: Path | None = None,
) -> None:
    """保存单文件 baseline 快照。"""
    root_path = _resolve_root(root_path)
    await _ensure_layout(root_path)
    payload = FileBaseline(
        path=Path(rel_path),
        content_hash=content_hash,
        content=content,
        symbols=symbols,
        captured_at=datetime.now(UTC),
    )
    await _atomic_write_json(
        _baseline_file_path(root_path, rel_path),
        payload.model_dump(mode="json"),
    )


async def load_file_baseline(
    rel_path: str | Path,
    root_path: Path | None = None,
) -> FileBaseline | None:
    """读取单文件 baseline 快照，不存在时返回 None。"""
    root_path = _resolve_root(root_path)
    path = _baseline_file_path(root_path, rel_path)
    if not path.exists():
        return None
    try:
        return FileBaseline.model_validate(await _read_json(path))
    except Exception:
        logger.exception(f"读取文件 baseline 失败: {path}")
        return None


async def delete_file_baseline(
    rel_path: str | Path,
    root_path: Path | None = None,
) -> None:
    """删除单文件 baseline 快照。"""
    root_path = _resolve_root(root_path)
    path = _baseline_file_path(root_path, rel_path)
    try:
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except Exception:
        logger.exception(f"删除文件 baseline 失败: {path}")
