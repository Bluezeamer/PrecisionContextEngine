from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import TraceEvent


class TraceStore:
    """轻量 JSONL trace 存储。

    第一版只做本地 JSONL 追加写入，满足 runtime 排障与 e2e 验证需要。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.base_dir = self.project_root / ".pce" / "v2" / "traces"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def new_request_id(self) -> str:
        return str(uuid.uuid4())

    def trace_path(self, request_id: str) -> Path:
        return self.base_dir / f"{request_id}.jsonl"

    def append(self, event: TraceEvent) -> None:
        path = self.trace_path(event.request_id)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def emit(self, *, request_id: str, mode: str, event: str, payload: dict[str, str] | None = None) -> None:
        self.append(
            TraceEvent(
                event=event,
                request_id=request_id,
                mode=mode,
                timestamp=datetime.now(timezone.utc),
                payload=payload or {},
            )
        )
