from __future__ import annotations

from pathlib import Path

from ..contracts import ReconcileDecision, ReconcileRequest, ReconcileResult
from ..engine.session import ProjectSession
from ..maintenance.file_state import fingerprint_file
from ..stores.baseline_store import BaselineStore


class ReconcileService:
    """v2 结构与认知失效协调器。

    第一版目标：
    - 识别 dirty files 的实质变更
    - 删除受影响的 insight 记录
    - 必要时触发导航树重建
    """

    def __init__(self) -> None:
        self._baseline_cache: dict[Path, BaselineStore] = {}

    def reconcile(self, session: ProjectSession, request: ReconcileRequest) -> ReconcileResult:
        baseline_store = self._baseline_store(session.project_root)
        baselines = baseline_store.load()

        changed_substantive: list[str] = []
        deleted_files: list[str] = []
        for rel_path in request.dirty_files:
            abs_path = session.project_root / rel_path
            current_fingerprint = fingerprint_file(abs_path)
            if current_fingerprint is None:
                deleted_files.append(rel_path)
                baseline_store.delete(rel_path)
                changed_substantive.append(rel_path)
                continue

            existing = baselines.get(rel_path)
            if existing is None or existing.fingerprint != current_fingerprint:
                baseline_store.upsert(rel_path, current_fingerprint)
                changed_substantive.append(rel_path)

        removed_insights = session.insight_store.delete_records_for_paths(changed_substantive)

        rebuilt_navigation = False
        decision = ReconcileDecision.NOOP
        affected_areas: list[str] = []
        update_plan = session.navigation_store.plan_update(changed_substantive, deleted_files)
        affected_areas = update_plan.affected_areas
        if update_plan.decision is ReconcileDecision.REBUILD_TREE:
            session.navigation_store.rebuild()
            rebuilt_navigation = True
            decision = ReconcileDecision.REBUILD_TREE
        elif update_plan.decision is ReconcileDecision.PATCH_TREE:
            session.navigation_store.patch_areas(update_plan.affected_areas)
            rebuilt_navigation = True
            decision = ReconcileDecision.PATCH_TREE

        if changed_substantive and decision is ReconcileDecision.NOOP:
            decision = ReconcileDecision.PATCH_TREE

        return ReconcileResult(
            decision=decision,
            changed_files=changed_substantive,
            removed_insights=removed_insights,
            rebuilt_navigation=rebuilt_navigation,
            affected_areas=affected_areas,
        )

    def seed_baselines(self, session: ProjectSession) -> int:
        baseline_store = self._baseline_store(session.project_root)
        tree = session.navigation_store.ensure_tree()
        count = 0
        for binding in tree.bindings:
            fingerprint = fingerprint_file(session.project_root / binding.file_path)
            if fingerprint is None:
                continue
            baseline_store.upsert(binding.file_path, fingerprint)
            count += 1
        return count

    def _baseline_store(self, project_root: Path) -> BaselineStore:
        project_root = project_root.resolve()
        store = self._baseline_cache.get(project_root)
        if store is None:
            store = BaselineStore(project_root)
            self._baseline_cache[project_root] = store
        return store
