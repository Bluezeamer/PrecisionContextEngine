from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import ConfidenceLevel, InsightHostRef, InsightRecord, InsightRecordMeta

_START_RE = re.compile(r"^<!--\s*PCE_INSIGHT\s+(\{.*\})\s*-->$", re.MULTILINE)
_END_MARKER = "<!-- /PCE_INSIGHT -->"


class InsightContainerStore:
    """Insight 宿主容器存储。

    宿主文件按 index / area / module 聚合。单条 insight 采用可索引记录块格式，
    便于 append-only 写入与按绑定文件删除整块。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.base_dir = self.project_root / ".pce" / "v2" / "insights"

    def ensure_layout(self) -> None:
        (self.base_dir / "areas").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "modules").mkdir(parents=True, exist_ok=True)

    def host_path(self, host: InsightHostRef) -> Path:
        return self.base_dir / host.relative_path

    def append(
        self,
        host: InsightHostRef,
        *,
        content: str,
        files: list[str],
        confidence: ConfidenceLevel | None = None,
        created_at: datetime | None = None,
        record_id: str | None = None,
    ) -> InsightRecord:
        self.ensure_layout()
        meta = InsightRecordMeta(
            id=record_id or str(uuid.uuid4()),
            files=[self._normalize_rel_path(item) for item in files if item.strip()],
            created_at=created_at or datetime.now(timezone.utc),
            confidence=confidence,
        )
        record = InsightRecord(meta=meta, content=content.strip())
        path = self.host_path(host)
        if path.exists():
            existing = path.read_text("utf-8").rstrip()
            text = f"{existing}\n\n{self._render_record(record)}\n"
        else:
            text = f"{self._render_header(host)}\n\n{self._render_record(record)}\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, "utf-8")
        return record

    def list_records(self, host: InsightHostRef) -> list[InsightRecord]:
        path = self.host_path(host)
        if not path.exists():
            return []
        return self._parse_records(path.read_text("utf-8"))

    def delete_records_for_paths(self, changed_files: list[str]) -> int:
        self.ensure_layout()
        normalized = {self._normalize_rel_path(item) for item in changed_files if item.strip()}
        if not normalized:
            return 0

        removed = 0
        candidate_paths = [
            self.base_dir / "index.md",
            *(self.base_dir / "areas").glob("*.md"),
            *(self.base_dir / "modules").glob("*.md"),
        ]
        for path in candidate_paths:
            if not path.exists():
                continue
            original = path.read_text("utf-8")
            header, records = self._split_header_and_records(original)
            kept: list[InsightRecord] = []
            for record in records:
                if normalized.intersection(record.meta.files):
                    removed += 1
                    continue
                kept.append(record)
            rewritten = header.rstrip() + ("\n\n" if kept else "\n")
            if kept:
                rewritten += "\n\n".join(self._render_record(item) for item in kept) + "\n"
            path.write_text(rewritten, "utf-8")
        return removed

    def needs_compaction(self, host: InsightHostRef, *, max_chars: int) -> bool:
        path = self.host_path(host)
        if not path.exists():
            return False
        return len(path.read_text("utf-8")) > max_chars

    def _split_header_and_records(self, content: str) -> tuple[str, list[InsightRecord]]:
        matches = list(_START_RE.finditer(content))
        if not matches:
            return content.rstrip(), []
        header = content[: matches[0].start()].rstrip()
        return header, self._parse_records(content)

    def _parse_records(self, content: str) -> list[InsightRecord]:
        records: list[InsightRecord] = []
        position = 0
        while True:
            match = _START_RE.search(content, position)
            if match is None:
                break
            meta = InsightRecordMeta.model_validate(json.loads(match.group(1)))
            body_start = match.end()
            body_end = content.find(_END_MARKER, body_start)
            if body_end == -1:
                break
            body = content[body_start:body_end].strip()
            records.append(InsightRecord(meta=meta, content=body))
            position = body_end + len(_END_MARKER)
        return records

    def _render_header(self, host: InsightHostRef) -> str:
        if host.kind.value == "index":
            title = "# PCE v2 Index Insights"
        elif host.kind.value == "area":
            title = f"# PCE v2 Area Insights · {host.slug}"
        else:
            title = f"# PCE v2 Module Insights · {host.slug}"
        return title + "\n\n> 自动生成的 insight 容器。默认 append-only，记录块由系统维护。"

    def _render_record(self, record: InsightRecord) -> str:
        meta_json = json.dumps(record.meta.model_dump(mode="json"), ensure_ascii=False)
        return f"<!-- PCE_INSIGHT {meta_json} -->\n{record.content}\n{_END_MARKER}"

    @staticmethod
    def _normalize_rel_path(value: str) -> str:
        text = Path(value).as_posix().replace("\\", "/").strip()
        while text.startswith("./"):
            text = text[2:]
        return text
