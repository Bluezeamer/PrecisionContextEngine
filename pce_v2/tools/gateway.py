from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import PreparedExecution, ToolDescriptor
from ..engine.session import ProjectSession
from ..integrations.serena import SerenaAdapter
from ..runtime.prompting import PromptAssembler
from ..retrieval.modes import ImpactMode, QueryMode


class ReadOnlyToolGateway:
    """v2 只读工具网关。

    第一版只提供 query/impact 需要的最小只读工具集，且明确与 insight/navigation 分离。
    """

    def __init__(self) -> None:
        self._assembler = PromptAssembler()

    def prepare_execution(self, session: ProjectSession, request) -> PreparedExecution:
        if request.mode.value == "query":
            tool_specs = QueryMode.tool_specs
        else:
            tool_specs = ImpactMode.tool_specs
        descriptors = [
            ToolDescriptor(
                name=item.name,
                purpose=item.purpose,
                result_policy=item.result_policy,
                read_only=item.read_only,
            )
            for item in tool_specs
        ]
        prompt = self._assembler.assemble(request)
        return PreparedExecution(request=request, prompt=prompt, tools=descriptors)

    def build_openai_tools(self, execution: PreparedExecution) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for descriptor in execution.tools:
            if descriptor.name == "navigation_read":
                tools.append(
                    self._tool_schema(
                        "navigation_read",
                        descriptor.purpose,
                        {"type": "object", "properties": {}, "additionalProperties": False},
                    )
                )
            elif descriptor.name == "insight_read":
                tools.append(
                    self._tool_schema(
                        "insight_read",
                        descriptor.purpose,
                        {
                            "type": "object",
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "相关文件相对路径列表",
                                }
                            },
                            "required": ["files"],
                            "additionalProperties": False,
                        },
                    )
                )
            elif descriptor.name == "code_search":
                tools.append(
                    self._tool_schema(
                        "code_search",
                        descriptor.purpose,
                        {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "relative_path": {"type": "string"},
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    )
                )
            elif descriptor.name == "code_read":
                tools.append(
                    self._tool_schema(
                        "code_read",
                        descriptor.purpose,
                        {
                            "type": "object",
                            "properties": {
                                "relative_path": {"type": "string"},
                                "max_lines": {"type": "integer", "minimum": 1},
                            },
                            "required": ["relative_path"],
                            "additionalProperties": False,
                        },
                    )
                )
            elif descriptor.name == "impact_graph":
                tools.append(
                    self._tool_schema(
                        "impact_graph",
                        descriptor.purpose,
                        {
                            "type": "object",
                            "properties": {
                                "symbol_name": {"type": "string"},
                                "relative_path": {"type": "string"},
                            },
                            "required": ["symbol_name", "relative_path"],
                            "additionalProperties": False,
                        },
                    )
                )
        tools.append(
            self._tool_schema(
                "deliver",
                "提交最终结构化结论并结束当前任务。",
                {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            )
        )
        return tools

    def read_navigation_summary(self, session: ProjectSession) -> str:
        tree = session.navigation_store.ensure_tree()
        area_count = sum(1 for item in tree.nodes if item.kind.value == "area")
        module_count = sum(1 for item in tree.nodes if item.kind.value == "module")
        return (
            f"tree={session.navigation_store.tree_path}\n"
            f"areas={area_count}\n"
            f"modules={module_count}\n"
            f"bindings={len(tree.bindings)}"
        )

    def read_insights_for_files(self, session: ProjectSession, files: list[str]) -> list[str]:
        host = session.navigation_store.resolve_host_for_files(files)
        records = session.insight_store.list_records(host)
        matched: list[str] = []
        target_set = {self._normalize(item) for item in files if item.strip()}
        for record in records:
            if target_set.intersection(record.meta.files):
                matched.append(record.content)
        return matched

    async def execute_tool(
        self,
        session: ProjectSession,
        *,
        name: str,
        arguments: dict[str, Any],
        max_result_chars: int,
        serena: SerenaAdapter | None = None,
    ) -> str:
        if name == "navigation_read":
            return self._clip(self.read_navigation_summary(session), max_result_chars)
        if name == "insight_read":
            records = self.read_insights_for_files(session, arguments.get("files", []))
            return self._clip("\n\n".join(records) or "(无相关 insight)", max_result_chars)
        if name == "code_search":
            result = await self.search_code(
                session,
                query=str(arguments.get("query", "")),
                relative_path=str(arguments.get("relative_path", "")),
                serena=serena,
            )
            return self._clip(json.dumps(result, ensure_ascii=False), max_result_chars)
        if name == "code_read":
            result = await self.read_code(
                session,
                relative_path=str(arguments.get("relative_path", "")),
                max_lines=int(arguments.get("max_lines", 200)),
                serena=serena,
            )
            return self._clip(result, max_result_chars)
        if name == "impact_graph":
            result = await self.read_impact_graph(
                session,
                symbol_name=str(arguments.get("symbol_name", "")),
                relative_path=str(arguments.get("relative_path", "")),
                serena=serena,
            )
            return self._clip(json.dumps(result, ensure_ascii=False), max_result_chars)
        raise ValueError(f"未知工具: {name}")

    async def search_code(
        self,
        session: ProjectSession,
        *,
        query: str,
        relative_path: str = "",
        serena: SerenaAdapter | None = None,
    ) -> Any:
        if serena is not None:
            return await serena.search_for_pattern(query, relative_path=relative_path)
        async with SerenaAdapter(session.project_root) as adapter:
            return await adapter.search_for_pattern(query, relative_path=relative_path)

    async def read_code(
        self,
        session: ProjectSession,
        *,
        relative_path: str,
        max_lines: int = 200,
        serena: SerenaAdapter | None = None,
    ) -> str:
        if serena is not None:
            return serena.read_file(relative_path, max_lines=max_lines)
        async with SerenaAdapter(session.project_root) as adapter:
            return adapter.read_file(relative_path, max_lines=max_lines)

    async def search_symbol(
        self,
        session: ProjectSession,
        *,
        pattern: str,
        relative_path: str = "",
        include_body: bool = False,
        serena: SerenaAdapter | None = None,
    ) -> Any:
        if serena is not None:
            return await serena.find_symbol(
                pattern,
                relative_path=relative_path,
                include_body=include_body,
                depth=1,
                substring_matching=True,
            )
        async with SerenaAdapter(session.project_root) as adapter:
            return await adapter.find_symbol(
                pattern,
                relative_path=relative_path,
                include_body=include_body,
                depth=1,
                substring_matching=True,
            )

    async def read_impact_graph(
        self,
        session: ProjectSession,
        *,
        symbol_name: str,
        relative_path: str,
        serena: SerenaAdapter | None = None,
    ) -> Any:
        if serena is not None:
            return await serena.find_referencing_symbols(symbol_name, relative_path)
        async with SerenaAdapter(session.project_root) as adapter:
            return await adapter.find_referencing_symbols(symbol_name, relative_path)

    @staticmethod
    def _normalize(value: str) -> str:
        text = Path(value).as_posix().replace("\\", "/").strip()
        while text.startswith("./"):
            text = text[2:]
        return text

    @staticmethod
    def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    @staticmethod
    def _clip(value: str, max_chars: int) -> str:
        if len(value) <= max_chars:
            return value
        return value[:max_chars].rstrip() + "\n\n...(truncated)"
