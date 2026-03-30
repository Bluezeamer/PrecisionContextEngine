"""Digest patch facts 构建器。"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import token as token_types
import tokenize
from io import StringIO
from pathlib import Path

from .memory import load_file_baseline, load_index
from .models import ChangedFileFact, PatchBlock, SymbolFact

logger = logging.getLogger(__name__)


class DigestDeltaBuilder:
    """为 digest stageB 构建最小可用的 patch facts。"""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    async def build_patch_facts(
        self,
        *,
        changed_files: list[str],
        deleted_files: list[str] | None = None,
    ) -> list[ChangedFileFact]:
        deleted = set(deleted_files or [])
        snapshot = await load_index(root_path=self.project_root)
        if snapshot is None:
            return []

        entries_map = {str(entry.file_meta.path): entry for entry in snapshot.entries}
        patch_facts: list[ChangedFileFact] = []
        for rel_path in [*changed_files, *deleted]:
            file_fact = await self._build_changed_file_fact(
                rel_path,
                current_entry=entries_map.get(rel_path),
                deleted=rel_path in deleted,
            )
            if file_fact is not None:
                patch_facts.append(file_fact)
        return patch_facts

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
