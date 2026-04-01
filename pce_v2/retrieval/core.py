from __future__ import annotations

from ..contracts import ImpactRequest, ModeName, PreparedRetrievalRequest, QueryRequest
from ..engine.session import ProjectSession
from .modes import ImpactMode, QueryMode


class RetrievalCore:
    """v2 检索内核的准备阶段。

    当前实现先落以下职责：
    - 基于 session 组装稳定上下文
    - 将 query / impact 显式映射为 mode 行为模板
    - 为后续 runtime / agent 执行层提供统一输入契约
    """

    def prepare_query(
        self,
        *,
        session: ProjectSession,
        request: QueryRequest,
    ) -> PreparedRetrievalRequest:
        session.touch()
        return PreparedRetrievalRequest(
            mode=ModeName.QUERY,
            session_stable_context=self._build_session_context(session),
            turn_local_context={
                "question": request.question,
                "navigation_tree_path": str(session.navigation_store.tree_path),
                "insight_root": str(session.insight_store.base_dir),
            },
            policy_context={
                "tool_mode": QueryMode.tool_policy.mode,
                "allowed_tools": ", ".join(QueryMode.tool_policy.allowed_tools),
                "denied_paths": ", ".join(QueryMode.tool_policy.denied_path_prefixes),
            },
            contract=QueryMode.contract,
            budget=QueryMode.budget,
        )

    def prepare_impact(
        self,
        *,
        session: ProjectSession,
        request: ImpactRequest,
    ) -> PreparedRetrievalRequest:
        session.touch()
        return PreparedRetrievalRequest(
            mode=ModeName.IMPACT,
            session_stable_context=self._build_session_context(session),
            turn_local_context={
                "target": request.target,
                "change_type": request.change_type,
                "file": request.file or "",
                "navigation_tree_path": str(session.navigation_store.tree_path),
                "insight_root": str(session.insight_store.base_dir),
            },
            policy_context={
                "tool_mode": ImpactMode.tool_policy.mode,
                "allowed_tools": ", ".join(ImpactMode.tool_policy.allowed_tools),
                "denied_paths": ", ".join(ImpactMode.tool_policy.denied_path_prefixes),
            },
            contract=ImpactMode.contract,
            budget=ImpactMode.budget,
        )

    def _build_session_context(self, session: ProjectSession) -> dict[str, str]:
        return {
            "project_root": str(session.project_root),
            "navigation_tree_exists": str(session.navigation_store.exists()).lower(),
            "navigation_tree_path": str(session.navigation_store.tree_path),
            "insight_root": str(session.insight_store.base_dir),
            "session_id": session.session_id,
        }
