"""模块级 DigestDelta 构建器。"""

from __future__ import annotations

import ast
import difflib
import hashlib
import logging
import re
import token as token_types
import tokenize
from io import StringIO
from pathlib import Path

from .insight_cache import InsightCache
from .memory import get_module_annotation, load_file_baseline, load_index
from .models import (
    ChangedFileFact,
    InsightFact,
    ModuleDigestDelta,
    ModuleRecord,
    ModuleRegistry,
    PatchBlock,
    SymbolFact,
)
from .module_registry import ModuleRegistryManager

logger = logging.getLogger(__name__)


class DigestDeltaBuilder:
    """构建模块级认知修正事实包。"""

    def __init__(self, project_root: Path, insight_cache: InsightCache) -> None:
        self.project_root = project_root.resolve()
        self.insight_cache = insight_cache
        self.registry = ModuleRegistryManager(self.project_root)

    async def build_for_changes(
        self,
        *,
        changed_files: list[str],
        deleted_files: list[str] | None = None,
    ) -> list[ModuleDigestDelta]:
        deleted = set(deleted_files or [])
        snapshot = await load_index(root_path=self.project_root)
        if snapshot is None:
            return []

        registry, current_file_to_record, historical_file_to_record = (
            await self.registry.build_file_owner_maps()
        )
        entries_map = {str(entry.file_meta.path): entry for entry in snapshot.entries}

        def _resolve_owner(path: str) -> ModuleRecord | None:
            return (
                current_file_to_record.get(path)
                or historical_file_to_record.get(path)
                or self._guess_deleted_owner(path, registry)
            )

        affected_module_ids: list[str] = []
        for path in [*changed_files, *deleted]:
            record = _resolve_owner(path)
            if record is not None and record.module_id not in affected_module_ids:
                affected_module_ids.append(record.module_id)

        insight_records = await self.insight_cache.get_all_records(include_stale=True)
        module_to_insights: dict[str, list[InsightFact]] = {}
        for record in insight_records:
            owner = _resolve_owner(record.scope)
            if owner is None:
                continue
            content = await self.insight_cache.get_entry_content(record.id)
            if not content:
                continue
            module_to_insights.setdefault(owner.module_id, []).append(
                InsightFact(
                    id=record.id,
                    scope=record.scope,
                    content=content,
                    confidence=record.confidence,
                    created_at=record.created_at,
                )
            )
            if owner.module_id not in affected_module_ids:
                affected_module_ids.append(owner.module_id)

        results: list[ModuleDigestDelta] = []
        for module_id in affected_module_ids:
            record = registry.records[module_id]
            module_file_facts: list[ChangedFileFact] = []
            for path in [*changed_files, *deleted]:
                owner = _resolve_owner(path)
                if owner is None or owner.module_id != module_id:
                    continue
                file_fact = await self._build_changed_file_fact(
                    path,
                    current_entry=entries_map.get(path),
                    deleted=path in deleted,
                )
                if file_fact is not None:
                    module_file_facts.append(file_fact)

            related_insights = module_to_insights.get(module_id, [])
            if not module_file_facts and not related_insights:
                continue

            results.append(
                ModuleDigestDelta(
                    module_id=record.module_id,
                    module_slug=record.slug,
                    module_name=record.display_name,
                    annotation_baseline=await get_module_annotation(
                        record.slug,
                        root_path=self.project_root,
                    )
                    or "",
                    related_insights=related_insights,
                    changed_files=module_file_facts,
                    change_scope_hint=self._classify_change_scope(module_file_facts),
                    external_context=[],
                )
            )
        return results

    @staticmethod
    def _classify_change_scope(
        changed_files: list[ChangedFileFact],
    ) -> str:
        if not changed_files:
            return "agent_decide"

        if all(file_fact.status == "modified" for file_fact in changed_files):
            return "module"

        created = [file_fact for file_fact in changed_files if file_fact.status == "created"]
        deleted = [file_fact for file_fact in changed_files if file_fact.status == "deleted"]
        modified = [file_fact for file_fact in changed_files if file_fact.status == "modified"]

        if created and deleted and not modified:
            remaining_deleted = list(deleted)
            matched = 0
            for created_fact in created:
                match_idx = next(
                    (
                        idx
                        for idx, deleted_fact in enumerate(remaining_deleted)
                        if created_fact.new_hash
                        and deleted_fact.old_hash
                        and created_fact.new_hash == deleted_fact.old_hash
                    ),
                    None,
                )
                if match_idx is None:
                    continue
                matched += 1
                remaining_deleted.pop(match_idx)
            if matched == len(created) and not remaining_deleted:
                return "route"

        return "agent_decide"

    @staticmethod
    def _guess_deleted_owner(path: str, registry: ModuleRegistry) -> ModuleRecord | None:
        """为缺失历史映射的 deleted path 提供轻量兜底归属。

        典型场景：
        - 技术路线迁移，旧目录整体替换为新目录（如 foo -> foo_v2）
        - 历史 registry 未完整覆盖，删除事件只能依赖路径相似性回挂模块
        """
        deleted_path = Path(path)
        best_record: ModuleRecord | None = None
        best_score = 0.0

        for record in registry.records.values():
            if record.status != "active":
                continue
            candidate_paths = {
                *record.file_paths,
                *record.historical_file_paths,
            }
            for candidate in candidate_paths:
                score = DigestDeltaBuilder._score_deleted_path_similarity(
                    deleted_path,
                    Path(candidate),
                )
                if score > best_score:
                    best_score = score
                    best_record = record

        return best_record if best_score >= 1.35 else None

    @staticmethod
    def _score_deleted_path_similarity(deleted_path: Path, candidate_path: Path) -> float:
        deleted_str = deleted_path.as_posix()
        candidate_str = candidate_path.as_posix()
        score = difflib.SequenceMatcher(None, deleted_str, candidate_str).ratio()
        deleted_stem = deleted_path.stem
        candidate_stem = candidate_path.stem

        if deleted_path.name == candidate_path.name:
            score += 0.8
        if deleted_stem == candidate_stem:
            score += 0.4
        score += difflib.SequenceMatcher(None, deleted_stem, candidate_stem).ratio() * 0.5
        if deleted_stem.startswith(candidate_stem) or candidate_stem.startswith(deleted_stem):
            score += 0.35

        deleted_parts = set(deleted_path.parts)
        candidate_parts = set(candidate_path.parts)
        if deleted_parts and candidate_parts:
            overlap = len(deleted_parts & candidate_parts)
            union = len(deleted_parts | candidate_parts)
            score += (overlap / union) * 0.5

        deleted_group = DigestDeltaBuilder._normalize_dir_group(deleted_path.parent)
        candidate_group = DigestDeltaBuilder._normalize_dir_group(candidate_path.parent)
        if deleted_group and candidate_group and deleted_group == candidate_group:
            score += 0.9

        return score

    @staticmethod
    def _normalize_dir_group(path: Path) -> str:
        raw = path.as_posix().strip("./")
        if not raw:
            return ""
        raw = re.sub(r"[_-]?v\d+\b", "", raw)
        raw = raw.replace("__", "_")
        return raw

    async def _build_changed_file_fact(
        self,
        rel_path: str,
        *,
        current_entry,
        deleted: bool,
    ) -> ChangedFileFact | None:
        baseline = await load_file_baseline(rel_path, root_path=self.project_root)
        old_content = baseline.content if baseline is not None else None
        old_hash = baseline.content_hash if baseline is not None else None
        old_symbols = baseline.symbols if baseline is not None else []

        abs_path = self.project_root / rel_path
        if deleted or not abs_path.exists():
            status = "deleted" if deleted else "modified"
            new_content = None
            new_hash = None
            new_symbols: list[SymbolFact] = []
        else:
            new_content = abs_path.read_text(encoding="utf-8")
            new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            new_symbols = (
                [
                    SymbolFact(
                        name=sym.name,
                        kind=sym.kind,
                        line_start=sym.line_start,
                        line_end=sym.line_end,
                    )
                    for sym in current_entry.symbols
                ]
                if current_entry is not None
                else []
            )
            status = "created" if baseline is None else "modified"

        if (
            status == "modified"
            and old_content is not None
            and new_content is not None
            and not self._is_digest_worthy_change(
                rel_path=rel_path,
                old_content=old_content,
                new_content=new_content,
                old_symbols=old_symbols,
                new_symbols=new_symbols,
            )
        ):
            logger.info("跳过低价值 digest 变更: %s", rel_path)
            return None

        patch_blocks = self._make_patch_blocks(old_content, new_content)
        return ChangedFileFact(
            path=Path(rel_path),
            status=status,
            old_hash=old_hash,
            new_hash=new_hash,
            old_content=old_content,
            new_content=new_content,
            old_symbols=old_symbols,
            new_symbols=new_symbols,
            patch_blocks=patch_blocks,
        )

    @staticmethod
    def _is_digest_worthy_change(
        *,
        rel_path: str,
        old_content: str,
        new_content: str,
        old_symbols: list[SymbolFact],
        new_symbols: list[SymbolFact],
    ) -> bool:
        """轻量判断变更是否值得进入 digest。

        目标：在不引入昂贵语义分析的前提下，过滤掉纯空白/纯注释类低价值改动。
        """
        if old_content == new_content:
            return False
        if not DigestDeltaBuilder._symbols_equal(old_symbols, new_symbols):
            return True

        suffix = Path(rel_path).suffix.lower()
        old_semantic = DigestDeltaBuilder._normalize_semantic_content(old_content, suffix)
        new_semantic = DigestDeltaBuilder._normalize_semantic_content(new_content, suffix)
        return old_semantic != new_semantic

    @staticmethod
    def _symbols_equal(old_symbols: list[SymbolFact], new_symbols: list[SymbolFact]) -> bool:
        def _key(symbol: SymbolFact) -> tuple[str, str, int, int]:
            return (
                symbol.name,
                symbol.kind.value,
                symbol.line_start,
                symbol.line_end,
            )

        return [_key(sym) for sym in old_symbols] == [_key(sym) for sym in new_symbols]

    @staticmethod
    def _normalize_semantic_content(content: str, suffix: str) -> str:
        if suffix == ".py":
            normalized = DigestDeltaBuilder._normalize_python_content(content)
        elif suffix in {".js", ".jsx", ".ts", ".tsx", ".vue", ".java", ".go", ".rs", ".c", ".cpp", ".h"}:
            normalized = DigestDeltaBuilder._normalize_c_like_content(content)
        else:
            normalized = content

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        return "\n".join(lines)

    @staticmethod
    def _normalize_python_content(content: str) -> str:
        content = DigestDeltaBuilder._strip_python_docstrings(content)
        pieces: list[str] = []
        try:
            stream = StringIO(content)
            for tok in tokenize.generate_tokens(stream.readline):
                if tok.type in {
                    token_types.INDENT,
                    token_types.DEDENT,
                    token_types.NEWLINE,
                    tokenize.NL,
                    tokenize.COMMENT,
                    token_types.ENDMARKER,
                }:
                    continue
                pieces.append(tok.string)
        except tokenize.TokenError:
            return content
        return " ".join(piece for piece in pieces if piece.strip())

    @staticmethod
    def _strip_python_docstrings(content: str) -> str:
        """剥离模块/类/函数 docstring，避免仅文档说明变更触发 digest。"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return content

        class _DocstringStripper(ast.NodeTransformer):
            @staticmethod
            def _strip_body(body: list[ast.stmt]) -> list[ast.stmt]:
                if not body:
                    return body
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(getattr(first, "value", None), ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    return body[1:]
                return body

            def visit_Module(self, node: ast.Module) -> ast.AST:
                self.generic_visit(node)
                node.body = self._strip_body(node.body)
                return node

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                self.generic_visit(node)
                node.body = self._strip_body(node.body)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
                self.generic_visit(node)
                node.body = self._strip_body(node.body)
                return node

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                self.generic_visit(node)
                node.body = self._strip_body(node.body)
                return node

        stripped = _DocstringStripper().visit(tree)
        ast.fix_missing_locations(stripped)
        try:
            return ast.unparse(stripped)
        except Exception:
            return content

    @staticmethod
    def _normalize_c_like_content(content: str) -> str:
        without_block = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
        without_html = re.sub(r"<!--.*?-->", "", without_block, flags=re.S)
        without_line = re.sub(r"//.*$", "", without_html, flags=re.M)
        return without_line

    @staticmethod
    def _make_patch_blocks(
        old_content: str | None,
        new_content: str | None,
    ) -> list[PatchBlock]:
        old_lines = (old_content or "").splitlines()
        new_lines = (new_content or "").splitlines()
        import difflib

        matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
        blocks: list[PatchBlock] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            blocks.append(
                PatchBlock(
                    old_start=i1 + 1 if i1 < i2 else None,
                    old_end=i2 if i1 < i2 else None,
                    new_start=j1 + 1 if j1 < j2 else None,
                    new_end=j2 if j1 < j2 else None,
                    old_snippet="\n".join(old_lines[i1:i2]),
                    new_snippet="\n".join(new_lines[j1:j2]),
                )
            )
        return blocks
