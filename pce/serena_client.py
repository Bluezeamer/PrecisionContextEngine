"""Serena MCP 客户端封装。

负责拉起 Serena 子进程、建立 MCP stdio 连接,并提供类型安全的工具调用接口。

使用方式:
    client = SerenaClient()
    await client.connect(project_path="/path/to/project", serena_install_path="/path/to/serena")
    result = await client.list_dir(".", recursive=False)
    await client.disconnect()

或使用上下文管理器:
    async with SerenaClient.create(project_path, serena_install_path) as client:
        result = await client.list_dir(".", recursive=False)
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
_PROXY_ENV_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
}

# Serena 写工具：修改代码库的操作，透传给上层 Agent 直接调用，PCE Agent 不参与推理
EDIT_TOOL_NAMES: set[str] = {
    "replace_symbol_body",
    "insert_after_symbol",
    "insert_before_symbol",
    "rename_symbol",
    "create_text_file",
    "replace_content",
}


def _collect_serena_env() -> dict[str, str]:
    """收集需要传递给 Serena 子进程的环境变量。"""
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("UV_") or key in _PROXY_ENV_KEYS:
            env[key] = value
    return env


# ============================================================================
# 异常定义
# ============================================================================


class SerenaClientError(RuntimeError):
    """Serena 客户端基础异常。"""


class SerenaConnectionError(SerenaClientError):
    """连接或初始化失败。"""


class SerenaToolError(SerenaClientError):
    """工具调用失败。"""


class SerenaTimeoutError(SerenaClientError):
    """工具调用或连接超时。"""


# ============================================================================
# 辅助函数
# ============================================================================


def _jsonable(value: Any) -> Any:
    """递归地将 MCP 响应对象转换为 JSON 可序列化的结构。

    Serena 工具通常在 CallToolResult.content[0].text 中返回 JSON 字符串,
    此函数会自动尝试解析,让调用方拿到 dict/list 而非原始字符串。
    """
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    # 字符串:尝试 JSON 解析,失败则原样返回
    if isinstance(value, str):
        try:
            import json
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    # Pydantic 模型
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    # MCP SDK 的响应对象(如 CallToolResult)
    # isError=True 时内容是错误信息,交由 call() 检查后统一处理
    if hasattr(value, "content"):
        content = value.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "text"):
                    parts.append(_jsonable(item.text))
                else:
                    parts.append(_jsonable(item))
            return parts if len(parts) > 1 else (parts[0] if parts else None)
        return _jsonable(content)
    # 通用 dataclass / 对象
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return str(value)


def _extract_tools(tools_result: Any) -> list[Any]:
    """从 list_tools() 的返回值中提取工具列表。"""
    if isinstance(tools_result, list):
        return tools_result
    if isinstance(tools_result, dict) and "tools" in tools_result:
        return tools_result["tools"]
    if hasattr(tools_result, "tools"):
        return list(tools_result.tools)
    raise SerenaClientError(f"无法解析 Serena 的工具列表: {type(tools_result)}")


def _tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """将 Serena 工具定义转换为 OpenAI function calling 格式。"""
    if isinstance(tool, dict):
        name = tool.get("name")
        description = tool.get("description") or ""
        parameters = tool.get("inputSchema") or tool.get("input_schema") or {}
    else:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", "") or ""
        # MCP SDK 使用 inputSchema 字段
        parameters = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", {})

    if not name:
        raise SerenaClientError(f"工具缺少 name 字段: {tool!r}")

    # 确保 parameters 是合法的 JSON Schema 对象
    if not parameters:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# ============================================================================
# 主客户端类
# ============================================================================


class SerenaClient:
    """Serena MCP 客户端封装。

    负责管理 Serena 子进程的生命周期和工具调用。
    """

    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._timeout_seconds = timeout_seconds
        self._session: ClientSession | None = None
        self._stdio_cm: Any | None = None
        self._session_cm: Any | None = None
        self._read_tools_schema: list[dict[str, Any]] = []
        self._edit_tools_schema: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()
        self._project_path: Path | None = None

    @property
    def connected(self) -> bool:
        """是否已建立连接。"""
        return self._session is not None

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        """返回只读工具的 schema 列表（OpenAI function calling 格式）。供 PCE Agent Loop 使用。"""
        return list(self._read_tools_schema)

    @property
    def edit_tools_schema(self) -> list[dict[str, Any]]:
        """返回写工具的 schema 列表（OpenAI function calling 格式）。供上层 Agent 透传调用。"""
        return list(self._edit_tools_schema)

    @property
    def project_path(self) -> Path:
        """返回当前项目路径。"""
        if self._project_path is None:
            raise SerenaClientError("SerenaClient 尚未连接,project_path 未初始化")
        return self._project_path

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        project_path: str | Path,
        serena_install_path: str | Path,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> AsyncIterator[SerenaClient]:
        """工厂方法,作为异步上下文管理器使用,自动管理连接生命周期。

        Usage:
            async with SerenaClient.create(project_path, serena_install_path) as client:
                result = await client.list_dir(".", recursive=False)
        """
        client = cls(timeout_seconds=timeout_seconds)
        await client.connect(project_path, serena_install_path)
        try:
            yield client
        finally:
            await client.disconnect()

    async def connect(
        self, project_path: str | Path, serena_install_path: str | Path
    ) -> None:
        """拉起 Serena 子进程并建立 MCP 连接,拉取工具 schema。

        Args:
            project_path: 目标项目根路径
            serena_install_path: Serena 的安装目录(含 pyproject.toml)

        Raises:
            SerenaConnectionError: 连接或初始化失败
            SerenaTimeoutError: 初始化超时
        """
        if self.connected:
            logger.debug("SerenaClient 已连接,跳过重复 connect")
            return

        self._project_path = Path(project_path).resolve()
        serena_path = Path(serena_install_path).resolve()

        server_params = StdioServerParameters(
            command="uv",
            args=[
                "run",
                "--project",
                str(serena_path),
                "serena-mcp-server",
                "--project",
                str(self._project_path),
            ],
            env=_collect_serena_env(),
        )

        try:
            # 建立 stdio 通道
            self._stdio_cm = stdio_client(server_params)
            read, write = await self._stdio_cm.__aenter__()

            # 建立 MCP session
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()

            # 初始化 MCP 握手
            await asyncio.wait_for(
                self._session.initialize(), timeout=self._timeout_seconds
            )

            # 一次性拉取并缓存所有工具 schema
            tools_result = await asyncio.wait_for(
                self._session.list_tools(), timeout=self._timeout_seconds
            )
            tools = _extract_tools(tools_result)
            all_schemas = [_tool_to_openai_schema(t) for t in tools]
            self._tool_names = {s["function"]["name"] for s in all_schemas}

            # 按职责分类：写工具（修改代码库）透传给上层，其余全部留给 PCE Agent
            read_tools: list[dict[str, Any]] = []
            edit_tools: list[dict[str, Any]] = []
            for schema in all_schemas:
                name = schema["function"]["name"]
                if name in EDIT_TOOL_NAMES:
                    edit_tools.append(schema)
                else:
                    read_tools.append(schema)

            self._read_tools_schema = read_tools
            self._edit_tools_schema = edit_tools

            logger.info(
                f"Serena 连接成功: project={self._project_path}, 工具数={len(self._tool_names)}"
            )

        except asyncio.TimeoutError as e:
            await self.disconnect()
            raise SerenaTimeoutError("Serena 初始化超时") from e
        except Exception as e:
            await self.disconnect()
            raise SerenaConnectionError(f"Serena 初始化失败: {e}") from e

    async def disconnect(self) -> None:
        """关闭连接并回收 Serena 子进程。"""
        # 关闭 session(顺序与 connect 相反)
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                logger.exception("关闭 MCP session 失败")
            finally:
                self._session_cm = None
                self._session = None

        # 关闭 stdio 通道(同时终止子进程)
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                logger.exception("关闭 stdio 连接失败")
            finally:
                self._stdio_cm = None

        self._read_tools_schema = []
        self._edit_tools_schema = []
        self._tool_names = set()
        logger.debug("SerenaClient 已断开连接")

    # ============================================================================
    # 统一调用接口
    # ============================================================================

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        """统一的工具调用入口,供 Agent Loop 在处理 LLM tool_call 时使用。

        Args:
            tool_name: Serena 工具名称
            args: 工具参数字典

        Returns:
            工具调用的 JSON 可序列化结果

        Raises:
            SerenaToolError: 工具调用失败或工具不存在
            SerenaTimeoutError: 调用超时
        """
        if not self.connected or self._session is None:
            raise SerenaToolError("Serena 未连接,请先调用 connect()")

        if tool_name not in self._tool_names:
            raise SerenaToolError(
                f"未知工具: {tool_name!r},可用工具: {sorted(self._tool_names)}"
            )

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, args),
                timeout=self._timeout_seconds,
            )
            # 检查 MCP 工具是否返回错误(isError 标记,SDK 不一定抛异常)
            if hasattr(result, "isError") and result.isError:
                error_text = _jsonable(result)
                raise SerenaToolError(f"工具返回错误: {tool_name}: {error_text}")
            return _jsonable(result)
        except asyncio.TimeoutError as e:
            raise SerenaTimeoutError(f"工具调用超时: {tool_name}") from e
        except (SerenaToolError, SerenaTimeoutError):
            raise
        except Exception as e:
            logger.exception(f"工具调用失败: {tool_name}, args={args}")
            raise SerenaToolError(f"工具调用失败: {tool_name}: {e}") from e

    # ============================================================================
    # 具名工具接口(提供参数提示,内部委托给 call)
    # ============================================================================

    async def list_dir(
        self,
        relative_path: str,
        *,
        recursive: bool = False,
        skip_ignored_files: bool = False,
    ) -> Any:
        """列出目录内容。"""
        return await self.call(
            "list_dir",
            {
                "relative_path": relative_path,
                "recursive": recursive,
                "skip_ignored_files": skip_ignored_files,
            },
        )

    async def get_symbols_overview(self, relative_path: str, *, depth: int = 0) -> Any:
        """获取文件的符号概览(类、函数等顶层定义)。"""
        return await self.call(
            "get_symbols_overview", {"relative_path": relative_path, "depth": depth}
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
        """查找符号定义。"""
        return await self.call(
            "find_symbol",
            {
                "name_path_pattern": name_path_pattern,
                "relative_path": relative_path,
                "include_body": include_body,
                "depth": depth,
                "substring_matching": substring_matching,
            },
        )

    async def find_referencing_symbols(
        self, name_path: str, relative_path: str
    ) -> Any:
        """查找引用指定符号的所有位置。"""
        return await self.call(
            "find_referencing_symbols",
            {"name_path": name_path, "relative_path": relative_path},
        )

    async def search_for_pattern(
        self,
        substring_pattern: str,
        *,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
    ) -> Any:
        """在代码中搜索正则模式。"""
        return await self.call(
            "search_for_pattern",
            {
                "substring_pattern": substring_pattern,
                "relative_path": relative_path,
                "context_lines_before": context_lines_before,
                "context_lines_after": context_lines_after,
            },
        )

    async def find_file(self, file_mask: str, relative_path: str = ".") -> Any:
        """按文件名 glob 模式查找文件。"""
        return await self.call(
            "find_file", {"file_mask": file_mask, "relative_path": relative_path}
        )

    # ============================================================================
    # 非 MCP 文件读取(用于 README、配置文件等非代码文件)
    # ============================================================================

    def read_file(self, relative_path: str, max_lines: int = 200) -> str:
        """直接读取非代码文件,不走 MCP 协议。

        适用于 README.md、docker-compose.yml、.env.example 等文件。
        代码文件应优先使用 find_symbol / get_symbols_overview 以节省 token。

        Args:
            relative_path: 相对于项目根目录的文件路径
            max_lines: 最大读取行数,超出时追加 truncated 提示

        Returns:
            文件内容字符串

        Raises:
            SerenaClientError: 文件不存在或读取失败
        """
        target_path = (self.project_path / relative_path).resolve()

        # 防止路径穿越到项目目录之外
        if not target_path.is_relative_to(self.project_path):
            raise SerenaClientError(
                f"路径穿越被拒绝: {relative_path!r} 超出项目根目录范围"
            )

        try:
            lines: list[str] = []
            with target_path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        total_lines = i + sum(1 for _ in f) + 1
                        lines.append(
                            f"\n... (已截断,文件共 {total_lines} 行,仅显示前 {max_lines} 行)"
                        )
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except FileNotFoundError as e:
            raise SerenaClientError(f"文件不存在: {relative_path}") from e
        except Exception as e:
            logger.exception(f"读取文件失败: {target_path}")
            raise SerenaClientError(f"读取文件失败: {relative_path}: {e}") from e
