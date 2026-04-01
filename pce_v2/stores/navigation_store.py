from __future__ import annotations

import json
from pathlib import Path

from ..contracts import (
    InsightHostKind,
    InsightHostRef,
    NavigationNode,
    NavigationNodeKind,
    NavigationTree,
    NavigationUpdatePlan,
    ReconcileDecision,
)
from ..navigation import NavigationTreeBuilder


class NavigationStore:
    """v2 导航树存储。

    当前只实现最小读写与“按绑定文件追溯宿主”能力。
    构建与增量修复逻辑后续单独落地。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.base_dir = self.project_root / ".pce" / "v2" / "navigation"
        self.tree_path = self.base_dir / "tree.json"
        self._builder = NavigationTreeBuilder()

    def ensure_layout(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.tree_path.exists()

    def load(self) -> NavigationTree | None:
        if not self.tree_path.exists():
            return None
        raw = self.tree_path.read_text("utf-8")
        return NavigationTree.model_validate(json.loads(raw))

    def save(self, tree: NavigationTree) -> None:
        self.ensure_layout()
        self.tree_path.write_text(
            json.dumps(tree.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            "utf-8",
        )

    def rebuild(self) -> NavigationTree:
        tree = self._builder.build(self.project_root)
        self.save(tree)
        return tree

    def ensure_tree(self) -> NavigationTree:
        tree = self.load()
        if tree is not None:
            return tree
        return self.rebuild()

    def resolve_host_for_files(self, files: list[str]) -> InsightHostRef:
        tree = self.ensure_tree()
        if tree is None or not files:
            return InsightHostRef(kind=InsightHostKind.INDEX)

        normalized = [self._normalize_rel_path(item) for item in files if item.strip()]
        if not normalized:
            return InsightHostRef(kind=InsightHostKind.INDEX)

        binding_map = {binding.file_path: binding.node_id for binding in tree.bindings}
        node_map = {node.id: node for node in tree.nodes}

        module_ids: list[str] = []
        for file_path in normalized:
            node_id = binding_map.get(file_path)
            if node_id is None:
                return InsightHostRef(kind=InsightHostKind.INDEX)
            node = node_map.get(node_id)
            if node is None:
                return InsightHostRef(kind=InsightHostKind.INDEX)
            if node.kind is NavigationNodeKind.MODULE:
                module_ids.append(node.id)
                continue
            if node.kind is NavigationNodeKind.AREA:
                return InsightHostRef(kind=InsightHostKind.AREA, slug=node.slug)
            return InsightHostRef(kind=InsightHostKind.INDEX)

        if len(set(module_ids)) == 1:
            module = node_map[module_ids[0]]
            return InsightHostRef(kind=InsightHostKind.MODULE, slug=module.slug)

        area_ids = {self._resolve_area_id(node_map, module_id) for module_id in module_ids}
        area_ids.discard(None)
        if len(area_ids) == 1:
            area = node_map[next(iter(area_ids))]
            return InsightHostRef(kind=InsightHostKind.AREA, slug=area.slug)

        return InsightHostRef(kind=InsightHostKind.INDEX)

    def plan_update(self, changed_files: list[str], deleted_files: list[str]) -> NavigationUpdatePlan:
        normalized_changed = [self._normalize_rel_path(item) for item in changed_files if item.strip()]
        normalized_deleted = [self._normalize_rel_path(item) for item in deleted_files if item.strip()]
        all_changed = sorted(set([*normalized_changed, *normalized_deleted]))
        if not all_changed:
            return NavigationUpdatePlan(decision=ReconcileDecision.NOOP, affected_areas=[], reasons=[])

        reasons: list[str] = []
        affected_areas = self._builder.extract_area_keys(all_changed)

        if normalized_deleted:
            reasons.append("存在删除文件，保守升级为重建导航树")
            return NavigationUpdatePlan(
                decision=ReconcileDecision.REBUILD_TREE,
                affected_areas=affected_areas,
                reasons=reasons,
            )

        for rel_path in all_changed:
            path = Path(rel_path)
            if len(path.parts) <= 1:
                reasons.append("根目录文件变更，保守升级为重建导航树")
                return NavigationUpdatePlan(
                    decision=ReconcileDecision.REBUILD_TREE,
                    affected_areas=affected_areas,
                    reasons=reasons,
                )

        reasons.append("仅局部 area 内发生结构变更，可执行子树 patch")
        return NavigationUpdatePlan(
            decision=ReconcileDecision.PATCH_TREE,
            affected_areas=affected_areas,
            reasons=reasons,
        )

    def patch_areas(self, area_keys: list[str]) -> NavigationTree:
        existing = self.ensure_tree()
        current_files = existing.bindings
        file_paths = [binding.file_path for binding in current_files]
        affected = set(area_keys)

        preserved_files: list[str] = []
        for file_path in file_paths:
            area = self._builder.extract_area_keys([file_path])[0]
            if area not in affected:
                preserved_files.append(file_path)

        rebuilt = self._builder.build(self.project_root)
        rebuilt_map = {binding.file_path: binding.file_path for binding in rebuilt.bindings}
        merged_files = sorted(set(preserved_files + list(rebuilt_map.keys())))
        merged_tree = self._builder.build_from_files(self.project_root, merged_files)
        self.save(merged_tree)
        return merged_tree

    def _resolve_area_id(self, node_map: dict[str, NavigationNode], module_id: str) -> str | None:
        node = node_map.get(module_id)
        if node is None:
            return None
        parent_id = node.parent_id
        if not parent_id:
            return None
        parent = node_map.get(parent_id)
        if parent is None:
            return None
        if parent.kind is NavigationNodeKind.AREA:
            return parent.id
        return None

    @staticmethod
    def _normalize_rel_path(value: str) -> str:
        text = Path(value).as_posix().replace("\\", "/").strip()
        while text.startswith("./"):
            text = text[2:]
        return text
