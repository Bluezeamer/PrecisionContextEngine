from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import FileBaselineRecord


class BaselineStore:
    """v2 文件基线存储。

    只维护“文件 -> 实质内容指纹”映射，用于 dirty file 失效判断。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.base_dir = self.project_root / ".pce" / "v2" / "baselines"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "files.json"

    def load(self) -> dict[str, FileBaselineRecord]:
        if not self.index_path.exists():
            return {}
        raw = self.index_path.read_text("utf-8")
        payload = json.loads(raw)
        return {
            key: FileBaselineRecord.model_validate(value)
            for key, value in payload.items()
        }

    def save(self, records: dict[str, FileBaselineRecord]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(records.items())
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")

    def upsert(self, file_path: str, fingerprint: str) -> FileBaselineRecord:
        records = self.load()
        record = FileBaselineRecord(
            file_path=file_path,
            fingerprint=fingerprint,
            updated_at=datetime.now(timezone.utc),
        )
        records[file_path] = record
        self.save(records)
        return record

    def delete(self, file_path: str) -> None:
        records = self.load()
        if file_path in records:
            records.pop(file_path)
            self.save(records)
