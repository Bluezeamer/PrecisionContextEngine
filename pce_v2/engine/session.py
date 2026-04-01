from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..stores.insight_store import InsightContainerStore
from ..stores.navigation_store import NavigationStore
from ..trace import TraceStore


@dataclass(slots=True)
class ProjectSession:
    project_root: Path
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_touched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    navigation_store: NavigationStore = field(init=False)
    insight_store: InsightContainerStore = field(init=False)
    trace_store: TraceStore = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        self.navigation_store = NavigationStore(self.project_root)
        self.insight_store = InsightContainerStore(self.project_root)
        self.trace_store = TraceStore(self.project_root)
        self.navigation_store.ensure_layout()
        self.insight_store.ensure_layout()
        self.navigation_store.ensure_tree()

    def touch(self) -> None:
        self.last_touched_at = datetime.now(timezone.utc)
