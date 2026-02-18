"""PCE MCP Server 入口。

注册 4 个 MCP 工具并路由到对应模块:
- pce_init   → indexer.build_index
- pce_query  → agent.PCEAgent.query
- pce_impact → agent.PCEAgent.impact
- pce_status → memory.get_status

启动方式:
    uv run pce serve
    或通过 Claude Code MCP 配置挂载
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .agent import PCEAgent
from .indexer import build_index
from .memory import get_status, index_exists
from .serena_client import SerenaClient, SerenaClientError

logger = logging.getLogger(__name__)


# ============================================================================
# 辅助函数
# ============================================================================


def _make_tool_description(summary: str, trigger: str, replaces: str) -> str:
    """拼装工具描述,遵循"明确替代关系、说明调用时机"的设计原则。"""
    return (
        f"{summary}\n"
        f"触发时机: {trigger}\n"
        f"替代操作: {replaces}"
    )


def _text_response(content: Any) -> list[TextContent]:
    """将任意数据序列化为 MCP TextContent 列表。"""
    return [TextContent(type="text", text=json.dumps(content, ensure_ascii=False, indent=2))]


def _error_response(error_code: str, message: str) -> list[TextContent]:
    """统一错误响应格式。"""
    return _text_response({
        "success": False,
        "error_code": error_code,
        "error_message": message,
    })


# ============================================================================
# 工具注册
# ============================================================================


def _build_tools() -> list[Tool]:
    """构建 4 个 MCP 工具定义。"""
    return [
        Tool(
            name="pce_init",
            description=_make_tool_description(
                summary="对目标项目进行初始化扫描,建立结构索引、引用索引和语义注解。后续所有查询依赖此索引。",
                trigger="项目首次使用 PCE 时调用,或代码结构发生重大变更后手动刷新。",
                replaces="替代手动 ls/find/ctags 扫描,以及逐文件阅读建立心智模型的过程。",
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "目标项目根路径(默认为 PCE_PROJECT_PATH 环境变量)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "是否强制重建索引(默认 false,已有索引时跳过)",
                    },
                },
            },
        ),
        Tool(
            name="pce_query",
            description=_make_tool_description(
                summary="以自然语言提问,PCE 返回经过推理的结构化答案,包括相关符号的位置和简要说明。",
                trigger="需要理解某个模块职责、查找函数定义、理解调用关系时调用。",
                replaces="替代传统的 ls + cat + grep 探索流程,无需上层 Agent 自行组合工具。",
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "自然语言问题,例如: '认证逻辑的入口在哪里?'",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "会话 ID(可选,用于多轮对话;不传时自动创建新会话)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="pce_impact",
            description=_make_tool_description(
                summary=(
                    "给定一个修改目标,返回完整的影响边界:所有直接引用点、类型依赖和建议修改顺序。"
                    "这是 PCE 最核心的工具,直接解决'最后一公里'问题。"
                ),
                trigger="上层 Agent 准备修改某个符号或文件之前调用,获取完整影响边界。",
                replaces="替代手动追踪引用链(Cmd+Shift+F / grep -r),消除 build 试错循环。",
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "修改目标,可以是符号名(如 UserSession)或文件路径",
                    },
                    "change_type": {
                        "type": "string",
                        "description": "变更类型: modify | rename | delete | add_field | change_signature",
                        "enum": ["modify", "rename", "delete", "add_field", "change_signature"],
                    },
                    "session_id": {
                        "type": "string",
                        "description": "会话 ID(可选)",
                    },
                    "file": {
                        "type": "string",
                        "description": "符号所在文件路径(可选,提供后可加速定位)",
                    },
                },
                "required": ["target", "change_type"],
            },
        ),
        Tool(
            name="pce_status",
            description=_make_tool_description(
                summary="返回当前项目的 PCE 状态:索引是否存在、建立时间、基本统计信息。",
                trigger="诊断或调试时调用,例如确认索引是否需要刷新。",
                replaces="替代手动检查 .pce/ 目录文件。",
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "项目根路径(可选,默认为启动时配置的项目路径)",
                    },
                },
            },
        ),
    ]


# ============================================================================
# 请求处理
# ============================================================================


class PCEContext:
    """持有 Server 的全局上下文,供工具处理器使用。"""

    def __init__(
        self,
        project_path: Path,
        serena_path: Path,
        serena_client: SerenaClient,
        agent: PCEAgent,
    ) -> None:
        self.project_path = project_path
        self.serena_path = serena_path
        self.serena_client = serena_client
        self.agent = agent

    async def ensure_connected(self) -> None:
        """确保 Serena 连接已建立。"""
        if not self.serena_client.connected:
            logger.info("重新连接 Serena...")
            await self.serena_client.connect(self.project_path, self.serena_path)

    async def handle_init(
        self, project_path: str | None, force: bool
    ) -> dict[str, Any]:
        """处理 pce_init 请求。"""
        target = Path(project_path).resolve() if project_path else self.project_path

        # 如果项目路径变更,重新连接 Serena
        if target != self.project_path:
            logger.info(f"切换项目路径: {self.project_path} → {target}")
            await self.serena_client.disconnect()
            self.project_path = target
            await self.serena_client.connect(self.project_path, self.serena_path)

        # 已有索引且非强制模式时跳过
        if not force and await index_exists(root_path=self.project_path):
            return {
                "success": True,
                "message": "索引已存在,如需刷新请传入 force=true",
                "project_path": str(self.project_path),
            }

        snapshot = await build_index(
            project_path=self.project_path,
            serena_client=self.serena_client,
            memory_root=self.project_path,
        )
        return {
            "success": True,
            "message": "索引构建完成",
            "stats": snapshot.build_stats.model_dump(mode="json"),
            "project_meta": snapshot.project_meta.model_dump(mode="json"),
        }

    async def handle_query(
        self, query: str, session_id: str | None
    ) -> dict[str, Any]:
        """处理 pce_query 请求。"""
        response = await self.agent.query(
            question=query,
            session_id=session_id,
            memory_root=self.project_path,
            serena_client=self.serena_client,
        )
        return response.model_dump(mode="json")

    async def handle_impact(
        self,
        target: str,
        change_type: str,
        session_id: str | None,
        file: str | None,
    ) -> dict[str, Any]:
        """处理 pce_impact 请求。"""
        # 将 file 参数附加到 target 描述中
        effective_target = f"{target} (file={file})" if file else target

        response = await self.agent.impact(
            target=effective_target,
            change_type=change_type,
            session_id=session_id,
            memory_root=self.project_path,
            serena_client=self.serena_client,
        )
        return response.model_dump(mode="json")

    async def handle_status(self, project_path: str | None) -> dict[str, Any]:
        """处理 pce_status 请求。"""
        root = Path(project_path).resolve() if project_path else self.project_path
        status = await get_status(root_path=root)
        return {
            **status,
            "initialized": await index_exists(root_path=root),
            "project_path": str(root),
        }


# ============================================================================
# 主服务函数
# ============================================================================


async def serve() -> None:
    """启动 PCE MCP Server (stdio 模式)。

    读取环境变量:
        PCE_PROJECT_PATH: 目标项目根路径(默认当前目录)
        SERENA_PATH:      Serena 安装路径(默认 ./serena)
        PCE_LOG_LEVEL:    日志级别(默认 INFO)
    """
    # 配置日志
    log_level = os.getenv("PCE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    project_path = Path(os.getenv("PCE_PROJECT_PATH", str(Path.cwd()))).resolve()
    serena_path = Path(os.getenv("SERENA_PATH", "serena")).resolve()

    logger.info(f"PCE Server 启动: project={project_path}, serena={serena_path}")

    # 初始化组件
    serena_client = SerenaClient()
    try:
        await serena_client.connect(project_path, serena_path)
    except Exception as e:
        logger.error(f"Serena 连接失败: {e},将在首次工具调用时重试")

    agent = PCEAgent()
    ctx = PCEContext(project_path, serena_path, serena_client, agent)

    # 创建并配置 MCP Server
    server = Server("pce")
    tools = _build_tools()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            await ctx.ensure_connected()

            if name == "pce_init":
                result = await ctx.handle_init(
                    project_path=arguments.get("project_path"),
                    force=bool(arguments.get("force", False)),
                )
                return _text_response(result)

            if name == "pce_query":
                query = arguments.get("query", "")
                if not query:
                    return _error_response("INVALID_INPUT", "query 参数不能为空")
                result = await ctx.handle_query(
                    query=query,
                    session_id=arguments.get("session_id"),
                )
                return _text_response(result)

            if name == "pce_impact":
                target = arguments.get("target", "")
                change_type = arguments.get("change_type", "")
                if not target or not change_type:
                    return _error_response("INVALID_INPUT", "target 和 change_type 参数不能为空")
                result = await ctx.handle_impact(
                    target=target,
                    change_type=change_type,
                    session_id=arguments.get("session_id"),
                    file=arguments.get("file"),
                )
                return _text_response(result)

            if name == "pce_status":
                result = await ctx.handle_status(arguments.get("project_path"))
                return _text_response(result)

            return _error_response("UNKNOWN_TOOL", f"未知工具: {name}")

        except SerenaClientError as e:
            logger.error(f"Serena 客户端错误: {e}")
            return _error_response("SERENA_ERROR", str(e))
        except Exception as e:
            logger.exception(f"工具调用异常: {name}")
            return _error_response("INTERNAL_ERROR", f"内部错误: {e}")

    # 运行 stdio Server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """CLI 入口点。"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
