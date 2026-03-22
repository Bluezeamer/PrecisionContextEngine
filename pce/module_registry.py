"""模块稳定 identity 注册表管理。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import ModuleRecord, ModuleRegistry


class ModuleRegistryManager:
    """管理 .pce/module_registry.json。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._path = self.project_root / ".pce" / "module_registry.json"

    @staticmethod
    def normalize_slug(display_name: str, *, file_paths: list[str] | None = None) -> str:
        """将展示名规范化为稳定 slug。"""
        base = display_name.strip().lower().replace("&", " and ")
        base = re.sub(r"[^a-z0-9]+", "-", base)
        base = re.sub(r"-+", "-", base).strip("-")
        if base:
            return base

        if file_paths:
            stem = Path(file_paths[0]).stem.lower().replace("_", "-").replace(".", "-")
            stem = re.sub(r"[^a-z0-9-]+", "-", stem)
            stem = re.sub(r"-+", "-", stem).strip("-")
            if stem:
                return stem

        return f"module-{uuid.uuid4().hex[:8]}"

    async def ensure_layout(self) -> None:
        def _ensure() -> bool:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return self._path.exists()

        exists = await asyncio.to_thread(_ensure)
        if exists:
            return
        text = json.dumps(ModuleRegistry().model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        await asyncio.to_thread(self._path.write_text, text, "utf-8")

    async def load(self) -> ModuleRegistry:
        await self.ensure_layout()
        try:
            raw = await asyncio.to_thread(self._path.read_text, "utf-8")
            return ModuleRegistry.model_validate(json.loads(raw))
        except Exception:
            return ModuleRegistry()

    async def save(self, registry: ModuleRegistry) -> None:
        await self.ensure_layout()
        text = json.dumps(registry.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
        tmp = self._path.parent / f".{self._path.name}.{uuid.uuid4().hex}.tmp"
        try:
            await asyncio.to_thread(tmp.write_text, text, "utf-8")
            await asyncio.to_thread(os.replace, tmp, self._path)
        finally:
            if tmp.exists():
                await asyncio.to_thread(tmp.unlink, missing_ok=True)

    @staticmethod
    def _dedupe_keep_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @classmethod
    def _merge_historical_paths(
        cls,
        *,
        existing_current: list[str],
        existing_historical: list[str],
        new_current: list[str],
    ) -> list[str]:
        """合并历史路径，保留删除/迁移前的归属锚点。"""
        return cls._dedupe_keep_order([*existing_historical, *existing_current, *new_current])

    @staticmethod
    def _score_match(
        record: ModuleRecord,
        file_paths: list[str],
        key_symbols: list[str],
    ) -> float:
        current_files = set(record.file_paths)
        target_files = set(file_paths)
        if not current_files or not target_files:
            file_score = 0.0
        else:
            overlap = len(current_files & target_files)
            union = len(current_files | target_files)
            file_score = overlap / union if union else 0.0

        current_symbols = set(record.key_symbols)
        target_symbols = set(key_symbols)
        if not current_symbols or not target_symbols:
            symbol_score = 0.0
        else:
            overlap = len(current_symbols & target_symbols)
            union = len(current_symbols | target_symbols)
            symbol_score = overlap / union if union else 0.0

        return (file_score * 0.8) + (symbol_score * 0.2)

    async def find_best_match(
        self,
        *,
        file_paths: list[str],
        key_symbols: list[str],
        min_score: float = 0.55,
    ) -> ModuleRecord | None:
        registry = await self.load()
        best: tuple[float, ModuleRecord] | None = None
        for record in registry.records.values():
            if record.status != "active":
                continue
            score = self._score_match(record, file_paths, key_symbols)
            if score < min_score:
                continue
            if best is None or score > best[0]:
                best = (score, record)
        return best[1] if best is not None else None

    async def get_or_create_module(
        self,
        *,
        display_name: str,
        file_paths: list[str],
        key_symbols: list[str],
    ) -> ModuleRecord:
        registry = await self.load()
        normalized_files = self._dedupe_keep_order(file_paths)
        normalized_symbols = self._dedupe_keep_order(key_symbols)
        matched = await self.find_best_match(
            file_paths=normalized_files,
            key_symbols=normalized_symbols,
        )
        now = datetime.now(UTC)

        if matched is not None:
            record = registry.records[matched.module_id]
            if display_name != record.display_name and display_name not in record.aliases:
                record.aliases.append(record.display_name)
                record.display_name = display_name
            record.historical_file_paths = self._merge_historical_paths(
                existing_current=record.file_paths,
                existing_historical=record.historical_file_paths,
                new_current=normalized_files,
            )
            record.file_paths = normalized_files
            record.key_symbols = normalized_symbols
            record.updated_at = now
            await self.save(registry)
            return record

        module_id = str(uuid.uuid4())
        slug = self.normalize_slug(display_name, file_paths=normalized_files)
        record = ModuleRecord(
            module_id=module_id,
            slug=slug,
            display_name=display_name,
            file_paths=normalized_files,
            historical_file_paths=normalized_files,
            key_symbols=normalized_symbols,
            created_at=now,
            updated_at=now,
            aliases=[],
            status="active",
        )
        registry.records[module_id] = record
        await self.save(registry)
        return record

    async def build_file_owner_maps(
        self,
    ) -> tuple[ModuleRegistry, dict[str, ModuleRecord], dict[str, ModuleRecord]]:
        """返回当前文件映射与历史文件映射。

        规则：
        - `current_map` 只包含当前 file_paths。
        - `historical_map` 包含 historical_file_paths 与 current file_paths。
        - 若同一路径同时存在于当前映射与历史映射，调用方应优先采用当前映射。
        """
        registry = await self.load()
        current_map: dict[str, ModuleRecord] = {}
        historical_map: dict[str, ModuleRecord] = {}

        for record in registry.records.values():
            if record.status != "active":
                continue
            for file_path in record.file_paths:
                current_map[file_path] = record
            for file_path in self._dedupe_keep_order(
                [*record.historical_file_paths, *record.file_paths]
            ):
                historical_map[file_path] = record

        return registry, current_map, historical_map
