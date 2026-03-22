"""模块级 DigestDelta 构建器。"""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path

from .insight_cache import InsightCache
from .memory import get_module_annotation, load_file_baseline, load_index
from .models import (
    ChangedFileFact,
    InsightFact,
    ModuleDigestDelta,
    ModuleRecord,
    ModuleRegistry,
    PatchBlock,
    SymbolFact,
)
from .module_registry import ModuleRegistryManager


class DigestDeltaBuilder:
    """构建模块级认知修正事实包。"""

    def __init__(self, project_root: Path, insight_cache: InsightCache) -> None:
        self.project_root = project_root.resolve()
        self.insight_cache = insight_cache
        self.registry = ModuleRegistryManager(self.project_root)

    async def build_for_changes(
        self,
        *,
        changed_files: list[str],
        deleted_files: list[str] | None = None,
    ) -> list[ModuleDigestDelta]:
        deleted = set(deleted_files or [])
        snapshot = await load_index(root_path=self.project_root)
        if snapshot is None:
            return []

        registry, current_file_to_record, historical_file_to_record = (
            await self.registry.build_file_owner_maps()
        )
        entries_map = {str(entry.file_meta.path): entry for entry in snapshot.entries}

        def _resolve_owner(path: str) -> ModuleRecord | None:
            return (
                current_file_to_record.get(path)
                or historical_file_to_record.get(path)
                or self._guess_deleted_owner(path, registry)
            )

        affected_module_ids: list[str] = []
        for path in [*changed_files, *deleted]:
            record = _resolve_owner(path)
            if record is not None and record.module_id not in affected_module_ids:
                affected_module_ids.append(record.module_id)

        insight_records = await self.insight_cache.get_all_records(include_stale=True)
        module_to_insights: dict[str, list[InsightFact]] = {}
        for record in insight_records:
            owner = _resolve_owner(record.scope)
            if owner is None:
                continue
            content = await self.insight_cache.get_entry_content(record.id)
            if not content:
                continue
            module_to_insights.setdefault(owner.module_id, []).append(
                InsightFact(
                    id=record.id,
                    scope=record.scope,
                    content=content,
                    confidence=record.confidence,
                    created_at=record.created_at,
                )
            )
            if owner.module_id not in affected_module_ids:
                affected_module_ids.append(owner.module_id)

        results: list[ModuleDigestDelta] = []
        for module_id in affected_module_ids:
            record = registry.records[module_id]
            module_file_facts: list[ChangedFileFact] = []
            for path in [*changed_files, *deleted]:
                owner = _resolve_owner(path)
                if owner is None or owner.module_id != module_id:
                    continue
                module_file_facts.append(
                    await self._build_changed_file_fact(
                        path,
                        current_entry=entries_map.get(path),
                        deleted=path in deleted,
                    )
                )

            results.append(
                ModuleDigestDelta(
                    module_id=record.module_id,
                    module_slug=record.slug,
                    module_name=record.display_name,
                    annotation_baseline=await get_module_annotation(
                        record.slug,
                        root_path=self.project_root,
                    )
                    or "",
                    related_insights=module_to_insights.get(module_id, []),
                    changed_files=module_file_facts,
                    external_context=[],
                )
            )
        return results

    @staticmethod
    def _guess_deleted_owner(path: str, registry: ModuleRegistry) -> ModuleRecord | None:
        """为缺失历史映射的 deleted path 提供轻量兜底归属。

        典型场景：
        - 技术路线迁移，旧目录整体替换为新目录（如 foo -> foo_v2）
        - 历史 registry 未完整覆盖，删除事件只能依赖路径相似性回挂模块
        """
        deleted_path = Path(path)
        best_record: ModuleRecord | None = None
        best_score = 0.0

        for record in registry.records.values():
            if record.status != "active":
                continue
            candidate_paths = {
                *record.file_paths,
                *record.historical_file_paths,
            }
            for candidate in candidate_paths:
                score = DigestDeltaBuilder._score_deleted_path_similarity(
                    deleted_path,
                    Path(candidate),
                )
                if score > best_score:
                    best_score = score
                    best_record = record

        return best_record if best_score >= 1.35 else None

    @staticmethod
    def _score_deleted_path_similarity(deleted_path: Path, candidate_path: Path) -> float:
        deleted_str = deleted_path.as_posix()
        candidate_str = candidate_path.as_posix()
        score = difflib.SequenceMatcher(None, deleted_str, candidate_str).ratio()
        deleted_stem = deleted_path.stem
        candidate_stem = candidate_path.stem

        if deleted_path.name == candidate_path.name:
            score += 0.8
        if deleted_stem == candidate_stem:
            score += 0.4
        score += difflib.SequenceMatcher(None, deleted_stem, candidate_stem).ratio() * 0.5
        if deleted_stem.startswith(candidate_stem) or candidate_stem.startswith(deleted_stem):
            score += 0.35

        deleted_parts = set(deleted_path.parts)
        candidate_parts = set(candidate_path.parts)
        if deleted_parts and candidate_parts:
            overlap = len(deleted_parts & candidate_parts)
            union = len(deleted_parts | candidate_parts)
            score += (overlap / union) * 0.5

        deleted_group = DigestDeltaBuilder._normalize_dir_group(deleted_path.parent)
        candidate_group = DigestDeltaBuilder._normalize_dir_group(candidate_path.parent)
        if deleted_group and candidate_group and deleted_group == candidate_group:
            score += 0.9

        return score

    @staticmethod
    def _normalize_dir_group(path: Path) -> str:
        raw = path.as_posix().strip("./")
        if not raw:
            return ""
        raw = re.sub(r"[_-]?v\d+\b", "", raw)
        raw = raw.replace("__", "_")
        return raw

    async def _build_changed_file_fact(
        self,
        rel_path: str,
        *,
        current_entry,
        deleted: bool,
    ) -> ChangedFileFact:
        baseline = await load_file_baseline(rel_path, root_path=self.project_root)
        old_content = baseline.content if baseline is not None else None
        old_hash = baseline.content_hash if baseline is not None else None
        old_symbols = baseline.symbols if baseline is not None else []

        abs_path = self.project_root / rel_path
        if deleted or not abs_path.exists():
            status = "deleted" if deleted else "modified"
            new_content = None
            new_hash = None
            new_symbols: list[SymbolFact] = []
        else:
            new_content = abs_path.read_text(encoding="utf-8")
            new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            new_symbols = (
                [
                    SymbolFact(
                        name=sym.name,
                        kind=sym.kind,
                        line_start=sym.line_start,
                        line_end=sym.line_end,
                    )
                    for sym in current_entry.symbols
                ]
                if current_entry is not None
                else []
            )
            status = "created" if baseline is None else "modified"

        patch_blocks = self._make_patch_blocks(old_content, new_content)
        return ChangedFileFact(
            path=Path(rel_path),
            status=status,
            old_hash=old_hash,
            new_hash=new_hash,
            old_content=old_content,
            new_content=new_content,
            old_symbols=old_symbols,
            new_symbols=new_symbols,
            patch_blocks=patch_blocks,
        )

    @staticmethod
    def _make_patch_blocks(
        old_content: str | None,
        new_content: str | None,
    ) -> list[PatchBlock]:
        old_lines = (old_content or "").splitlines()
        new_lines = (new_content or "").splitlines()
        import difflib

        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
        blocks: list[PatchBlock] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            blocks.append(
                PatchBlock(
                    old_start=i1 + 1 if i1 < i2 else None,
                    old_end=i2 if i1 < i2 else None,
                    new_start=j1 + 1 if j1 < j2 else None,
                    new_end=j2 if j1 < j2 else None,
                    old_snippet="\n".join(old_lines[i1:i2]),
                    new_snippet="\n".join(new_lines[j1:j2]),
                )
            )
        return blocks
