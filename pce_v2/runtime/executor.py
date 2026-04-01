from __future__ import annotations

from ..contracts import ImpactRequest, PreparedExecution, QueryRequest
from ..engine.session import ProjectSession
from ..retrieval.core import RetrievalCore
from ..tools.gateway import ReadOnlyToolGateway


class QueryImpactExecutor:
    """v2 query/impact 执行准备器。

    当前阶段不直接执行 LLM loop，而是产出完整的执行准备面：
    - PreparedRetrievalRequest
    - AssembledPrompt
    - Tool descriptors
    - 初始导航摘要
    """

    def __init__(self) -> None:
        self._core = RetrievalCore()
        self._gateway = ReadOnlyToolGateway()

    def prepare_query(self, session: ProjectSession, request: QueryRequest) -> PreparedExecution:
        prepared = self._core.prepare_query(session=session, request=request)
        return self._gateway.prepare_execution(session, prepared)

    def prepare_impact(self, session: ProjectSession, request: ImpactRequest) -> PreparedExecution:
        prepared = self._core.prepare_impact(session=session, request=request)
        return self._gateway.prepare_execution(session, prepared)

    def materialize(self, session: ProjectSession, execution: PreparedExecution) -> dict[str, object]:
        return {
            "mode": execution.request.mode.value,
            "system_prompt": execution.prompt.system,
            "context_blocks": execution.prompt.context_blocks,
            "tools": [tool.model_dump(mode="json") for tool in execution.tools],
            "navigation_summary": self._gateway.read_navigation_summary(session),
            "budget": execution.request.budget.model_dump(mode="json"),
        }
