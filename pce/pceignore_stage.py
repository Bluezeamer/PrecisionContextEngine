"""init stage1: 生成 `.pce/pceignore` 的轻量认知阶段。"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .init_cognition_limits import PCEIGNORE_STAGE_MAX_ATTEMPTS
from .memory import _atomic_write_text
from .file_discovery import is_ignored, is_probably_text_file, is_hard_skipped
from .serena_client import SerenaClient
from .topology_cognition_agent import TopologyCognitionAgent

logger = logging.getLogger(__name__)

_HARD_PATTERNS = [".pce/", ".serena/"]
_MAX_FACT_TREE_DEPTH = 2
_MAX_FACT_TREE_LINES = 160
_MAX_STATS_ROWS = 48
_MAX_BINARY_ROWS = 24
_MAX_DIRTY_PATHS = 80
_MAX_DIRTY_GROUPS = 32


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _read_gitignore_text(project_root: Path) -> str:
    path = project_root / ".gitignore"
    if not path.exists():
        return ""
    try:
        return path.read_text("utf-8")
    except Exception:
        return ""


def _read_gitignore_patterns(project_root: Path) -> list[str]:
    text = _read_gitignore_text(project_root)
    patterns: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return _dedupe_keep_order(patterns)


def _read_existing_pceignore_patterns(project_root: Path) -> list[str]:
    path = project_root / ".pce" / "pceignore"
    if not path.exists():
        return []
    try:
        text = path.read_text("utf-8")
    except Exception:
        return []
    patterns: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in _HARD_PATTERNS:
            continue
        patterns.append(line)
    return _dedupe_keep_order(patterns)


def _iter_visible_files(project_root: Path) -> tuple[list[str], list[str]]:
    visible_dirs: set[str] = set()
    visible_files: list[str] = []
    for current_root, dirnames, filenames in os.walk(project_root, topdown=True):
        current_path = Path(current_root)
        rel_dir = current_path.relative_to(project_root).as_posix() if current_path != project_root else "."
        keep_dirs: list[str] = []
        for dirname in dirnames:
            rel_path = dirname if rel_dir == "." else f"{rel_dir}/{dirname}"
            if is_hard_skipped(rel_path) or is_ignored(project_root, rel_path):
                continue
            keep_dirs.append(dirname)
            visible_dirs.add(rel_path)
        dirnames[:] = keep_dirs

        for filename in filenames:
            rel_path = filename if rel_dir == "." else f"{rel_dir}/{filename}"
            if is_hard_skipped(rel_path) or is_ignored(project_root, rel_path):
                continue
            visible_files.append(rel_path)
    return sorted(visible_dirs), sorted(visible_files)


def _build_tree_lines(
    visible_dirs: list[str],
    visible_files: list[str],
    *,
    max_depth: int,
    max_lines: int,
) -> list[str]:
    lines: list[str] = []
    omitted = 0

    for rel_dir in visible_dirs:
        parts = Path(rel_dir).parts
        if len(parts) > max_depth:
            continue
        indent = "  " * (len(parts) - 1)
        lines.append(f"{indent}- {parts[-1]}/")
        if len(lines) >= max_lines:
            omitted += 1
            break

    if len(lines) < max_lines:
        remaining = max_lines - len(lines)
        for rel_file in visible_files:
            parts = Path(rel_file).parts
            if len(parts) > max_depth + 1:
                continue
            indent = "  " * (len(parts) - 1)
            lines.append(f"{indent}- {parts[-1]}")
            if len(lines) >= max_lines:
                omitted += 1
                break

    total_candidates = sum(1 for d in visible_dirs if len(Path(d).parts) <= max_depth) + sum(
        1 for f in visible_files if len(Path(f).parts) <= max_depth + 1
    )
    omitted += max(0, total_candidates - len(lines) - omitted)
    if omitted > 0:
        lines.append(f"[truncated: omitted {omitted} lines]")
    return lines


def _detect_kind(project_root: Path, rel_path: str) -> str:
    abs_path = project_root / rel_path
    try:
        return "text" if is_probably_text_file(abs_path) else "binary"
    except Exception:
        return "unknown"


def _top_exts(items: list[str], *, limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for item in items:
        suffix = Path(item).suffix.lower() or "(no-ext)"
        counter[suffix] += 1
    return [f"{ext}:{count}" for ext, count in counter.most_common(limit)]


def _build_directory_stats(project_root: Path, visible_dirs: list[str], visible_files: list[str]) -> list[dict[str, Any]]:
    bucket: dict[str, dict[str, Any]] = {}
    candidates = [".", *visible_dirs]
    for rel_dir in candidates:
        bucket[rel_dir] = {
            "path": rel_dir,
            "total_files": 0,
            "text_files": 0,
            "binary_files": 0,
            "sample_files": [],
            "extensions": [],
        }

    files_by_dir: dict[str, list[str]] = {key: [] for key in bucket}
    for rel_file in visible_files:
        parent = Path(rel_file).parent.as_posix()
        if parent == ".":
            parent = "."
        if parent not in files_by_dir:
            files_by_dir[parent] = []
            bucket[parent] = {
                "path": parent,
                "total_files": 0,
                "text_files": 0,
                "binary_files": 0,
                "sample_files": [],
                "extensions": [],
            }
        files_by_dir[parent].append(rel_file)

    rows: list[dict[str, Any]] = []
    for rel_dir, files in files_by_dir.items():
        if not files:
            continue
        text_count = 0
        binary_count = 0
        for rel_file in files:
            kind = _detect_kind(project_root, rel_file)
            if kind == "text":
                text_count += 1
            elif kind == "binary":
                binary_count += 1
        rows.append({
            "path": rel_dir,
            "total_files": len(files),
            "text_files": text_count,
            "binary_files": binary_count,
            "sample_files": files[:3],
            "extensions": _top_exts(files),
        })
    rows.sort(key=lambda item: (-item["total_files"], item["path"]))
    return rows[:_MAX_STATS_ROWS]


def _build_binary_clusters(project_root: Path, visible_files: list[str]) -> list[dict[str, Any]]:
    by_dir: dict[str, list[str]] = {}
    for rel_file in visible_files:
        if _detect_kind(project_root, rel_file) != "binary":
            continue
        parent = Path(rel_file).parent.as_posix()
        if parent == ".":
            parent = "."
        by_dir.setdefault(parent, []).append(rel_file)
    rows = [
        {"path": path, "binary_files": len(files), "sample_files": files[:3]}
        for path, files in by_dir.items()
    ]
    rows.sort(key=lambda item: (-item["binary_files"], item["path"]))
    return rows[:_MAX_BINARY_ROWS]


def _build_pceignore_facts(project_root: Path) -> dict[str, Any]:
    visible_dirs, visible_files = _iter_visible_files(project_root)
    return {
        "project_root": str(project_root),
        "gitignore": _read_gitignore_text(project_root),
        "tree_depth": _MAX_FACT_TREE_DEPTH,
        "tree_snapshot": "\n".join(
            _build_tree_lines(
                visible_dirs,
                visible_files,
                max_depth=_MAX_FACT_TREE_DEPTH,
                max_lines=_MAX_FACT_TREE_LINES,
            )
        ),
        "directory_stats": _build_directory_stats(project_root, visible_dirs, visible_files),
        "binary_clusters": _build_binary_clusters(project_root, visible_files),
    }


def _build_dirty_parent_groups(project_root: Path, dirty_paths: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rel_path in dirty_paths:
        normalized = str(rel_path).strip().lstrip("./")
        if not normalized:
            continue
        parent = Path(normalized).parent.as_posix()
        if parent == ".":
            parent = "."
        grouped[parent].append(normalized)
    rows = [
        {
            "path": path,
            "dirty_count": len(paths),
            "sample_files": sorted(paths)[:5],
        }
        for path, paths in grouped.items()
    ]
    rows.sort(key=lambda item: (-item["dirty_count"], item["path"]))
    return rows[:_MAX_DIRTY_GROUPS]


def _build_pceignore_refresh_facts(
    project_root: Path,
    *,
    changed_files: list[str],
    deleted_files: list[str],
) -> dict[str, Any]:
    base = _build_pceignore_facts(project_root)
    dirty_changed = _dedupe_keep_order([str(item).strip() for item in changed_files if str(item).strip()])
    dirty_deleted = _dedupe_keep_order([str(item).strip() for item in deleted_files if str(item).strip()])
    base.update({
        "mode": "refresh",
        "current_pceignore_patterns": _read_existing_pceignore_patterns(project_root),
        "dirty_changed_files": dirty_changed[:_MAX_DIRTY_PATHS],
        "dirty_deleted_files": dirty_deleted[:_MAX_DIRTY_PATHS],
        "dirty_parent_groups": _build_dirty_parent_groups(project_root, dirty_changed),
    })
    return base


def _normalize_ignore_patterns(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("ignore_patterns")
    if not isinstance(raw, list):
        raise ValueError("ignore_patterns 必须是字符串数组")
    normalized: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("ignore_patterns 必须全部为字符串")
        text = item.strip()
        if not text:
            continue
        if "\n" in text or "\x00" in text:
            raise ValueError(f"存在非法 pattern: {text!r}")
        normalized.append(text)
    return _dedupe_keep_order(normalized)


def _build_retry_feedback(exc: Exception, *, attempt: int, max_attempts: int) -> str:
    return "\n".join([
        f"上一次 `pceignore` 输出未通过校验（第 {attempt}/{max_attempts} 次尝试）。",
        f"错误：{type(exc).__name__}: {exc}",
        "请不要继续探索；请基于现有证据重新输出严格符合 schema 的 JSON：{'ignore_patterns':[...]}。",
    ])


async def _write_pceignore(project_root: Path, ignore_patterns: list[str]) -> None:
    merged = _dedupe_keep_order([
        *_HARD_PATTERNS,
        *_read_gitignore_patterns(project_root),
        *ignore_patterns,
    ])
    content = "\n".join([
        "# Auto-generated by PCE stage1",
        *merged,
        "",
    ])
    await _atomic_write_text(project_root / ".pce" / "pceignore", content)


def _normalize_refresh_payload(payload: dict[str, Any]) -> tuple[str, list[str], str]:
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"no_update", "append_patterns"}:
        raise ValueError("refresh payload.action 必须是 no_update 或 append_patterns")
    rationale = " ".join(str(payload.get("rationale") or "").strip().split())
    patterns = _normalize_ignore_patterns(payload)
    if action == "no_update" and patterns:
        raise ValueError("action=no_update 时 ignore_patterns 必须为空")
    if action == "append_patterns" and not patterns:
        raise ValueError("action=append_patterns 时 ignore_patterns 不能为空")
    return action, patterns, rationale


def _build_refresh_retry_feedback(exc: Exception, *, attempt: int, max_attempts: int) -> str:
    return "\n".join([
        f"上一次 `pceignore_refresh` 输出未通过校验（第 {attempt}/{max_attempts} 次尝试）。",
        f"错误：{type(exc).__name__}: {exc}",
        "若无需更新，请输出 `{action:\"no_update\", ignore_patterns:[], rationale:\"...\"}`。",
        "若需要更新，请输出 `{action:\"append_patterns\", ignore_patterns:[...], rationale:\"...\"}`，且 ignore_patterns 必须是最小增量。",
    ])


async def run_pceignore_stage(
    project_root: Path,
    serena_client: SerenaClient,
    *,
    model: str | None = None,
) -> None:
    target = project_root / ".pce" / "pceignore"
    if target.exists():
        return

    facts = _build_pceignore_facts(project_root)
    agent = TopologyCognitionAgent(
        project_root=project_root,
        discovery_facts=facts,
        model=model,
        max_seconds=120.0,
    )
    messages = agent.build_initial_messages()

    last_exc: Exception | None = None
    for attempt in range(1, PCEIGNORE_STAGE_MAX_ATTEMPTS + 1):
        logger.info(
            "pceignore stage start: attempt=%d/%d",
            attempt,
            PCEIGNORE_STAGE_MAX_ATTEMPTS,
        )
        try:
            payload = await agent.run_stage(
                stage="pceignore",
                messages=messages,
                serena_client=serena_client,
            )
            patterns = _normalize_ignore_patterns(payload)
            await _write_pceignore(project_root, patterns)
            logger.info("已生成 .pce/pceignore: %d 条规则", len(patterns))
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= PCEIGNORE_STAGE_MAX_ATTEMPTS:
                break
            messages.append({
                "role": "user",
                "content": _build_retry_feedback(
                    exc,
                    attempt=attempt,
                    max_attempts=PCEIGNORE_STAGE_MAX_ATTEMPTS,
                ),
            })

    logger.warning("pceignore stage 失败，降级写入最小规则: %s", last_exc)
    await _write_pceignore(project_root, [])


async def run_pceignore_refresh_stage(
    project_root: Path,
    serena_client: SerenaClient,
    *,
    changed_files: list[str],
    deleted_files: list[str] | None = None,
    model: str | None = None,
) -> str:
    target = project_root / ".pce" / "pceignore"
    if not target.exists():
        await run_pceignore_stage(project_root, serena_client, model=model)
        return "rebuilt_missing"
    dirty_changed = _dedupe_keep_order(changed_files)
    dirty_deleted = _dedupe_keep_order(deleted_files or [])
    if not dirty_changed and not dirty_deleted:
        return "no_dirty"

    facts = _build_pceignore_refresh_facts(
        project_root,
        changed_files=dirty_changed,
        deleted_files=dirty_deleted,
    )
    agent = TopologyCognitionAgent(
        project_root=project_root,
        discovery_facts=facts,
        model=model,
        max_seconds=120.0,
    )
    messages = agent.build_initial_messages()

    last_exc: Exception | None = None
    for attempt in range(1, PCEIGNORE_STAGE_MAX_ATTEMPTS + 1):
        logger.info(
            "pceignore refresh stage start: attempt=%d/%d changed=%d deleted=%d",
            attempt,
            PCEIGNORE_STAGE_MAX_ATTEMPTS,
            len(dirty_changed),
            len(dirty_deleted),
        )
        try:
            payload = await agent.run_stage(
                stage="pceignore_refresh",
                messages=messages,
                serena_client=serena_client,
            )
            action, patterns, rationale = _normalize_refresh_payload(payload)
            if action == "no_update":
                logger.info("pceignore refresh: no_update%s", f" ({rationale})" if rationale else "")
                return "no_update"
            await _write_pceignore(project_root, patterns)
            logger.info(
                "pceignore refresh: appended %d patterns%s",
                len(patterns),
                f" ({rationale})" if rationale else "",
            )
            return "updated"
        except Exception as exc:
            last_exc = exc
            if attempt >= PCEIGNORE_STAGE_MAX_ATTEMPTS:
                break
            messages.append({
                "role": "user",
                "content": _build_refresh_retry_feedback(
                    exc,
                    attempt=attempt,
                    max_attempts=PCEIGNORE_STAGE_MAX_ATTEMPTS,
                ),
            })

    logger.warning("pceignore refresh stage 失败，保持现有规则不变: %s", last_exc)
    return "failed_no_change"
