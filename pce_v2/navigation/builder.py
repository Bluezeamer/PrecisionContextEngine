from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..contracts import CoverageRule, FileBinding, NavigationNode, NavigationNodeKind, NavigationTree
from .discovery import discover_trackable_files

_ROOT_ID = "root"
_ROOT_SLUG = "root"
_ROOT_AREA_SLUG = "root"
_ROOT_MODULE_SLUG = "root_files"


class NavigationTreeBuilder:
    """v2 导航树构建器。

    第一版目标很克制：
    - 先建立稳定的 index -> area -> module 层级
    - 规则覆盖优先，文件唯一归属
    - 目录结构优先，不尝试做深层语义模块推断
    """

    def build(self, project_root: Path, *, files: list[str] | None = None) -> NavigationTree:
        root = project_root.resolve()
        file_list = files if files is not None else discover_trackable_files(root)
        return self.build_from_files(root, file_list)

    def build_from_files(self, project_root: Path, files: list[str]) -> NavigationTree:
        root = project_root.resolve()
        file_list = sorted(set(files))

        nodes: list[NavigationNode] = [
            NavigationNode(
                id=_ROOT_ID,
                kind=NavigationNodeKind.ROOT,
                name=root.name,
                slug=_ROOT_SLUG,
                parent_id=None,
            )
        ]
        rules: list[CoverageRule] = []
        bindings: list[FileBinding] = []

        area_groups = self._group_by_area(file_list)
        for area_key, area_files in sorted(area_groups.items()):
            area_slug = self._slugify(area_key)
            area_id = f"area:{area_slug}"
            area_name = area_key if area_key != _ROOT_AREA_SLUG else "root"
            nodes.append(
                NavigationNode(
                    id=area_id,
                    kind=NavigationNodeKind.AREA,
                    name=area_name,
                    slug=area_slug,
                    parent_id=_ROOT_ID,
                )
            )
            rules.append(
                CoverageRule(
                    node_id=area_id,
                    include_paths=self._build_area_include_paths(area_key, area_files),
                    exclude_paths=[],
                    priority=100,
                )
            )

            module_groups = self._group_by_module(area_key, area_files)
            for module_key, module_files in sorted(module_groups.items()):
                module_slug = self._module_slug(area_slug, module_key)
                module_id = f"module:{module_slug}"
                module_name = module_key if module_key != _ROOT_MODULE_SLUG else "root_files"
                nodes.append(
                    NavigationNode(
                        id=module_id,
                        kind=NavigationNodeKind.MODULE,
                        name=module_name,
                        slug=module_slug,
                        parent_id=area_id,
                    )
                )
                rules.append(
                    CoverageRule(
                        node_id=module_id,
                        include_paths=self._build_module_include_paths(area_key, module_key, module_files),
                        exclude_paths=[],
                        priority=200,
                    )
                )
                for file_path in sorted(module_files):
                    bindings.append(FileBinding(file_path=file_path, node_id=module_id))

        return NavigationTree(root_path=root, nodes=nodes, rules=rules, bindings=bindings)

    def extract_area_keys(self, files: list[str]) -> list[str]:
        area_keys: set[str] = set()
        for file_path in files:
            parts = Path(file_path).parts
            area_keys.add(parts[0] if len(parts) > 1 else _ROOT_AREA_SLUG)
        return sorted(area_keys)

    def _group_by_area(self, files: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for file_path in files:
            parts = Path(file_path).parts
            area_key = parts[0] if len(parts) > 1 else _ROOT_AREA_SLUG
            grouped[area_key].append(file_path)
        return dict(grouped)

    def _group_by_module(self, area_key: str, files: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for file_path in files:
            parts = Path(file_path).parts
            if area_key == _ROOT_AREA_SLUG:
                grouped[_ROOT_MODULE_SLUG].append(file_path)
                continue
            module_key = parts[1] if len(parts) > 2 else _ROOT_MODULE_SLUG
            grouped[module_key].append(file_path)
        return dict(grouped)

    def _build_area_include_paths(self, area_key: str, area_files: list[str]) -> list[str]:
        if area_key == _ROOT_AREA_SLUG:
            return sorted(area_files)
        return [f"{area_key}/"]

    def _build_module_include_paths(
        self,
        area_key: str,
        module_key: str,
        module_files: list[str],
    ) -> list[str]:
        if area_key == _ROOT_AREA_SLUG:
            return sorted(module_files)
        if module_key == _ROOT_MODULE_SLUG:
            root_level_files = [
                file_path
                for file_path in module_files
                if len(Path(file_path).parts) == 2
            ]
            return sorted(root_level_files)
        return [f"{area_key}/{module_key}/"]

    def _module_slug(self, area_slug: str, module_key: str) -> str:
        if area_slug == _ROOT_AREA_SLUG:
            return _ROOT_MODULE_SLUG
        return f"{area_slug}__{self._slugify(module_key)}"

    @staticmethod
    def _slugify(value: str) -> str:
        text = value.strip().replace("\\", "/")
        text = text.replace("/", "_")
        text = text.replace(" ", "_")
        return text.replace("-", "_") or "unknown"
