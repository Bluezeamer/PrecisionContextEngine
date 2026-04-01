from __future__ import annotations

import uuid
from pathlib import Path

from ..contracts import (
    ImpactRequest,
    PreparedExecution,
    PreparedRetrievalRequest,
    QueryRequest,
    ReconcileRequest,
    ReconcileResult,
)
from ..maintenance import ReconcileService
from ..retrieval.core import RetrievalCore
from ..runtime import MinimalReActRuntime, QueryImpactExecutor
from .session import ProjectSession


class PCEngine:
    """v2 项目会话与检索编排入口。

    第一批实现只负责：
    - 维护多项目 session
    - 提供 query / impact 的准备阶段
    - 隔离后续 runtime、tooling、agent 实现
    """

    def __init__(self) -> None:
        self.engine_id = str(uuid.uuid4())
        self._sessions: dict[Path, ProjectSession] = {}
        self._retrieval = RetrievalCore()
        self._executor = QueryImpactExecutor()
        self._runtime = MinimalReActRuntime()
        self._reconcile = ReconcileService()

    def bind(self, project_root: Path) -> ProjectSession:
        project_root = project_root.resolve()
        session = self._sessions.get(project_root)
        if session is None:
            session = ProjectSession(project_root=project_root)
            self._sessions[project_root] = session
        session.touch()
        return session

    def prepare_query(self, project_root: Path, request: QueryRequest) -> PreparedRetrievalRequest:
        session = self.bind(project_root)
        return self._retrieval.prepare_query(session=session, request=request)

    def prepare_impact(self, project_root: Path, request: ImpactRequest) -> PreparedRetrievalRequest:
        session = self.bind(project_root)
        return self._retrieval.prepare_impact(session=session, request=request)

    def prepare_query_execution(self, project_root: Path, request: QueryRequest) -> dict[str, object]:
        session = self.bind(project_root)
        execution = self._executor.prepare_query(session, request)
        return self._executor.materialize(session, execution)

    def prepare_impact_execution(self, project_root: Path, request: ImpactRequest) -> dict[str, object]:
        session = self.bind(project_root)
        execution = self._executor.prepare_impact(session, request)
        return self._executor.materialize(session, execution)

    def build_query_execution(self, project_root: Path, request: QueryRequest) -> PreparedExecution:
        session = self.bind(project_root)
        return self._executor.prepare_query(session, request)

    def build_impact_execution(self, project_root: Path, request: ImpactRequest) -> PreparedExecution:
        session = self.bind(project_root)
        return self._executor.prepare_impact(session, request)

    async def run_query(self, project_root: Path, request: QueryRequest) -> dict[str, object]:
        session = self.bind(project_root)
        return await self._runtime.run_query(session, request)

    async def run_impact(self, project_root: Path, request: ImpactRequest) -> dict[str, object]:
        session = self.bind(project_root)
        return await self._runtime.run_impact(session, request)

    def reconcile(self, project_root: Path, request: ReconcileRequest) -> ReconcileResult:
        session = self.bind(project_root)
        return self._reconcile.reconcile(session, request)

    def seed_baselines(self, project_root: Path) -> int:
        session = self.bind(project_root)
        return self._reconcile.seed_baselines(session)

    def list_sessions(self) -> list[ProjectSession]:
        return sorted(self._sessions.values(), key=lambda item: item.last_touched_at, reverse=True)
