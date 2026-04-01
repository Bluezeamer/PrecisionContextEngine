from __future__ import annotations

from pathlib import Path
from typing import Any

from pce.serena_client import SerenaClient


class SerenaAdapter:
    """v2 对 Serena 的窄封装。

    只复用底层 MCP 客户端，不复用旧 agent/business 逻辑。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._client = SerenaClient()

    async def __aenter__(self) -> "SerenaAdapter":
        await self._client.connect(self.project_root)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._client.disconnect()

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        return self._client.tools_schema

    async def list_dir(self, relative_path: str = ".", *, recursive: bool = False) -> Any:
        return await self._client.list_dir(relative_path, recursive=recursive)

    async def find_file(self, file_mask: str, relative_path: str = ".") -> Any:
        return await self._client.find_file(file_mask, relative_path)

    async def search_for_pattern(
        self,
        substring_pattern: str,
        *,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
    ) -> Any:
        return await self._client.search_for_pattern(
            substring_pattern,
            relative_path=relative_path,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
        )

    async def find_symbol(
        self,
        name_path_pattern: str,
        *,
        relative_path: str = "",
        include_body: bool = False,
        depth: int = 0,
        substring_matching: bool = False,
    ) -> Any:
        return await self._client.find_symbol(
            name_path_pattern,
            relative_path=relative_path,
            include_body=include_body,
            depth=depth,
            substring_matching=substring_matching,
        )

    async def find_referencing_symbols(self, name_path: str, relative_path: str) -> Any:
        return await self._client.find_referencing_symbols(name_path, relative_path)

    async def get_symbols_overview(self, relative_path: str, *, depth: int = 0) -> Any:
        return await self._client.get_symbols_overview(relative_path, depth=depth)

    def read_file(self, relative_path: str, max_lines: int = 200) -> str:
        return self._client.read_file(relative_path, max_lines=max_lines)
