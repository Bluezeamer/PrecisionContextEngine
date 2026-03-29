"""文件发现与 ignore 规则。

目标：
1. 以 ignore 黑名单为主导，而不是扩展名白名单。
2. baseline / staging 覆盖所有未被忽略的有效文本文件。
3. 符号索引仅作为增强层，只对少量已知语言文件启用。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pathspec

from .serena_language_registry import supports_symbol_index_for_path

HARD_SKIP_DIRS = frozenset(
    {
        ".git",
        ".pce",
        ".serena",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
    }
)

PCE_IGNORE_REL_PATH = Path(".pce/pceignore")
_TEXT_SNIFF_BYTES = 8192


def _normalize_rel_path(path: str | Path) -> str:
    return Path(path).as_posix().lstrip("./")


def is_hard_skipped(path: str | Path) -> bool:
    """是否命中内建硬规则目录。"""
    rel = Path(path)
    return any(part in HARD_SKIP_DIRS for part in rel.parts)


def supports_symbol_index(path: str | Path) -> bool:
    """是否启用符号索引增强。"""
    return supports_symbol_index_for_path(path)


@lru_cache(maxsize=32)
def _load_ignore_spec_cached(ignore_path_str: str, mtime_ns: int | None) -> pathspec.PathSpec | None:
    del mtime_ns
    ignore_path = Path(ignore_path_str)
    if not ignore_path.exists():
        return None
    lines = ignore_path.read_text("utf-8").splitlines()
    patterns = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not patterns:
        return None
    return pathspec.GitIgnoreSpec.from_lines(patterns)


def _load_ignore_spec(ignore_path: Path) -> pathspec.PathSpec | None:
    stat = ignore_path.stat() if ignore_path.exists() else None
    mtime_ns = stat.st_mtime_ns if stat is not None else None
    return _load_ignore_spec_cached(str(ignore_path.resolve()), mtime_ns)


def _matches_ignore(spec: pathspec.PathSpec | None, rel_path: str | Path) -> bool:
    if spec is None:
        return False
    normalized = _normalize_rel_path(rel_path)
    return spec.match_file(normalized)


def _matches_ignore_from_base(
    spec: pathspec.PathSpec | None,
    *,
    base_dir: Path,
    rel_path: str | Path,
) -> bool:
    if spec is None:
        return False
    normalized = _normalize_rel_path(rel_path)
    try:
        scoped = Path(normalized).relative_to(base_dir.as_posix())
    except ValueError:
        return False
    scoped_text = scoped.as_posix()
    return bool(scoped_text) and spec.match_file(scoped_text)


def is_ignored_by_project_gitignore(project_root: Path, rel_path: str | Path) -> bool:
    """检查项目树上从根到目标路径沿途的 `.gitignore`。"""
    root = project_root.resolve()
    normalized = _normalize_rel_path(rel_path)
    rel = Path(normalized)

    candidate_dirs = [root]
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        candidate_dirs.append(current)

    for base_dir in candidate_dirs:
        spec = _load_ignore_spec(base_dir / ".gitignore")
        if _matches_ignore_from_base(spec, base_dir=base_dir.relative_to(root), rel_path=normalized):
            return True
    return False


def is_ignored_by_pce_ignore(project_root: Path, rel_path: str | Path) -> bool:
    """检查 PCE 内部 ignore 黑名单。"""
    spec = _load_ignore_spec(project_root.resolve() / PCE_IGNORE_REL_PATH)
    return _matches_ignore(spec, rel_path)


def is_ignored(project_root: Path, rel_path: str | Path) -> bool:
    """综合 ignore 判断。

    优先级：
    1. 项目根 .gitignore
    2. PCE 内部 ignore 黑名单
    3. 内建硬规则目录
    """
    if is_hard_skipped(rel_path):
        return True
    if is_ignored_by_project_gitignore(project_root, rel_path):
        return True
    if is_ignored_by_pce_ignore(project_root, rel_path):
        return True
    return False


def is_visible_to_agent(project_root: Path, rel_path: str | Path) -> bool:
    """该路径是否应暴露给受控 Agent 读取/感知。

    规则保持与项目 ignore 体系一致：
    - 命中硬跳过目录不可见
    - 命中 `.gitignore` 不可见
    - 命中 `.pce/pceignore` 不可见
    """
    normalized = _normalize_rel_path(rel_path)
    if not normalized:
        return False
    return not is_ignored(project_root, normalized)


def is_probably_text_file(path: Path) -> bool:
    """基于文件前缀做轻量文本检测。"""
    try:
        chunk = path.read_bytes()[:_TEXT_SNIFF_BYTES]
    except Exception:
        return False

    if not chunk:
        return True
    if b"\x00" in chunk:
        return False

    # 低控制字符密度过高时，视作二进制内容。
    control_bytes = sum(
        1
        for byte in chunk
        if byte < 32 and byte not in {9, 10, 12, 13}
    )
    return (control_bytes / len(chunk)) <= 0.30


def should_track_existing_file(project_root: Path, rel_path: str | Path) -> bool:
    """是否应跟踪当前存在的文件。"""
    if is_ignored(project_root, rel_path):
        return False
    abs_path = project_root.resolve() / _normalize_rel_path(rel_path)
    if not abs_path.exists() or not abs_path.is_file():
        return False
    return is_probably_text_file(abs_path)


def should_track_deleted_path(rel_path: str | Path) -> bool:
    """删除事件的保守策略：只排除内建硬规则目录。"""
    return not is_hard_skipped(rel_path)


def filter_trackable_files(project_root: Path, file_paths: list[str]) -> list[str]:
    """过滤出应纳入 baseline / 索引视野的文本文件。"""
    results: list[str] = []
    seen: set[str] = set()
    for rel_path in file_paths:
        normalized = _normalize_rel_path(rel_path)
        if normalized in seen:
            continue
        if not should_track_existing_file(project_root, normalized):
            continue
        seen.add(normalized)
        results.append(normalized)
    return sorted(results)


def filter_visible_paths(project_root: Path, paths: list[str]) -> list[str]:
    """过滤出允许暴露给 Agent 的相对路径。"""
    results: list[str] = []
    seen: set[str] = set()
    for rel_path in paths:
        normalized = _normalize_rel_path(rel_path)
        if not normalized or normalized in seen:
            continue
        if not is_visible_to_agent(project_root, normalized):
            continue
        seen.add(normalized)
        results.append(normalized)
    return results
