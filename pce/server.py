"""PCE MCP Server 入口。

注册 MCP 工具并路由到对应模块:
- pce_query  → agent.PCEAgent.query (首次调用时自动构建索引)
- pce_impact → agent.PCEAgent.impact
- pce_status → memory.get_status
- pce_sync   → SerenaClient 断开重连刷新索引
- pce_replace_symbol_body / pce_insert_after_symbol /
  pce_insert_before_symbol / pce_rename_symbol
              → SerenaClient.call() 直接透传写操作

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

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .agent import PCEAgent
from .insight_cache import InsightCache
from .indexer import build_index, build_index_incremental
from .staging import DirtyState, FileWatcher, StagingArea
from .memory import get_status, index_exists
from .serena_client import (
    DEFAULT_TIMEOUT_SECONDS,
    SerenaClient,
    SerenaClientError,
    EDIT_TOOL_NAMES,
)

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


def _get_serena_timeout() -> int:
    raw = os.getenv("PCE_SERENA_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"PCE_SERENA_TIMEOUT 值非法: {raw},使用默认值 {DEFAULT_TIMEOUT_SECONDS}")
        return DEFAULT_TIMEOUT_SECONDS
    if value <= 0:
        logger.warning(f"PCE_SERENA_TIMEOUT 值需为正数: {raw},使用默认值 {DEFAULT_TIMEOUT_SECONDS}")
        return DEFAULT_TIMEOUT_SECONDS
    return value


# ============================================================================
# 工具注册
# ============================================================================



def _extract_affected_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """从写工具参数中提取受影响的文件相对路径。"""
    # Serena 写工具统一使用 relative_path 参数
    return arguments.get("relative_path")


def _build_edit_tools(edit_tools_schema: list[dict[str, Any]]) -> list[Tool]:
    """将 Serena 写工具 schema 转换为 PCE MCP 工具定义，加 pce_ 前缀。"""
    tools: list[Tool] = []
    for schema in edit_tools_schema:
        fn = schema.get("function", {})
        name = fn.get("name")
        if not name or name not in EDIT_TOOL_NAMES:
            continue
        tools.append(
            Tool(
                name=f"pce_{name}",
                description=fn.get("description") or "",
                inputSchema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return tools


def _build_tools(edit_tools_schema: list[dict[str, Any]] | None = None) -> list[Tool]:
    """构建 MCP 工具定义（含写工具透传与索引同步信号）。"""
    base_tools = [
        Tool(
            name="pce_query",
            description=_make_tool_description(
                summary=(
                    "以自然语言提问,PCE 返回经过推理的结构化答案。"
                    "可在 query 中指定输出格式,例如要求返回 {file, line_range, name_path} 的符号定位列表,"
                    "或 file:line 格式的精确位置——尤其适合在执行代码修改前定位手术靶点。"
                ),
                trigger=(
                    "需要理解模块职责、查找函数/类定义、理解调用关系时调用。"
                    "也应在准备修改某段代码前使用,用于精确定位目标符号的 name_path 与行号范围。"
                ),
                replaces=(
                    "替代传统 ls + cat + grep 的多步探索链;"
                    "PCE 在独立上下文内完成全部检索与推理,不消耗也不污染上层 Agent 的对话上下文。"
                ),
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
                    "给定修改目标,返回完整影响边界:所有直接引用点、类型依赖与建议修改顺序。"
                    "可在 target 字段末尾追加格式要求,例如'请以结构化列表返回每处引用点的行号、name_path 和代码片段',"
                    "便于上层 Agent 直接落地修改而无需二次检索。"
                ),
                trigger="上层 Agent 准备修改某个符号或文件之前调用,获取完整影响边界与精确修改位置。",
                replaces=(
                    "替代手动追踪引用链(Cmd+Shift+F / grep -r)与反复 build 试错;"
                    "多步检索与推理在 PCE 独立上下文完成,结果以结构化形式一次性交付给上层 Agent。"
                ),
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

    # 写工具：从 Serena edit_tools_schema 动态生成
    edit_tools = _build_edit_tools(edit_tools_schema or [])

    # 索引同步信号
    sync_tool = Tool(
        name="pce_sync",
        description=_make_tool_description(
            summary="通知 PCE 代码库已发生变更，触发 Serena 重连并重建 PCE 索引。后续 pce_query/pce_impact 将看到最新代码。",
            trigger="上层 Agent 完成一批代码修改后调用，确保 PCE 索引与代码库同步。",
            replaces="替代手动重启 PCE 服务或等待索引自动失效。",
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    )

    return [*base_tools, *edit_tools, sync_tool]


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
        insight_cache: InsightCache,
    ) -> None:
        self.project_path = project_path
        self.serena_path = serena_path
        self.serena_client = serena_client
        self.agent = agent
        self.insight_cache = insight_cache
        self._sync_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._initialized = False
        # bootstrap 状态：eager 初始化完成信号 + 过程中收集的 warning
        self._bootstrap_event = asyncio.Event()
        self._bootstrap_warnings: list[str] = []
        # 文件变更暂存区与监听
        self.staging = StagingArea(project_path)
        self.watcher = FileWatcher(self.staging)

    async def ensure_connected(self) -> None:
        """确保 Serena 连接已建立。"""
        if not self.serena_client.connected:
            logger.info("重新连接 Serena...")
            await self.serena_client.connect(self.project_path, self.serena_path)

    async def _run_index_refresh(self) -> None:
        """执行索引刷新逻辑（全量或增量）。

        调用方负责加锁（_sync_lock）与状态位更新。
        1. 索引不存在 → 全量构建 + 清空暂存区
        2. 索引存在但暂存区有累积变更 → 增量更新 + 清空暂存区
        3. 索引存在且无变更 → 直接跳过
        """
        if not await index_exists(root_path=self.project_path):
            logger.info("PCE 索引不存在，开始全量构建...")
            dirty = await self.staging.list_pending_reindex()
            all_paths = dirty.changed + dirty.deleted
            hash_snapshot = (
                await self.staging.snapshot_hashes(all_paths)
                if all_paths
                else None
            )
            await build_index(
                project_path=self.project_path,
                serena_client=self.serena_client,
                memory_root=self.project_path,
            )
            if all_paths:
                await self.staging.acknowledge_after_reindex(
                    all_paths, expected_hashes=hash_snapshot,
                )
            logger.info("PCE 索引全量构建完成")
            return

        dirty = await self.staging.list_pending_reindex()
        if not dirty.empty:
            all_paths = dirty.changed + dirty.deleted
            hash_snapshot = await self.staging.snapshot_hashes(all_paths)
            logger.info(
                f"检测到 {len(dirty.changed)} 个变更、"
                f"{len(dirty.deleted)} 个删除，执行增量索引更新"
            )
            await build_index_incremental(
                project_path=self.project_path,
                serena_client=self.serena_client,
                memory_root=self.project_path,
                changed_files=dirty.changed,
                deleted_files=dirty.deleted,
            )
            await self.staging.acknowledge_after_reindex(
                all_paths, expected_hashes=hash_snapshot
            )
            logger.info("增量索引更新完成")

    async def _bootstrap(self) -> None:
        """后台 bootstrap 任务：校验 Serena 项目激活 + 构建索引。

        不依赖 LLM，纯程序化。失败仅记录 warning，不阻塞 event 置位。
        pce_query / pce_impact 通过 await _bootstrap_event.wait() 等待此任务完成。
        """
        if self._bootstrap_event.is_set():
            return

        async with self._init_lock:
            if self._bootstrap_event.is_set():
                return

            try:
                async with self._sync_lock:
                    # 1. 显式校验 Serena 项目激活
                    #    Serena 启动参数已带 --project，但启动时激活失败会被静默吞掉。
                    #    再调一次 activate_project 可捕获激活失败并记录 warning。
                    try:
                        result = await self.serena_client.call(
                            "activate_project",
                            {"project": str(self.project_path)},
                        )
                        logger.info("Serena 项目激活校验成功: %s", result)
                    except Exception as e:
                        warning = (
                            f"Serena 项目激活校验失败: {e}。"
                            "符号级索引可能不完整，请检查 .serena/project.yml 配置。"
                        )
                        self._bootstrap_warnings.append(warning)
                        logger.warning(warning)

                    # 2. 构建索引
                    try:
                        await self._run_index_refresh()
                    except Exception as e:
                        warning = f"Bootstrap 索引构建失败: {e}"
                        self._bootstrap_warnings.append(warning)
                        logger.warning(warning)
            except Exception as e:
                self._bootstrap_warnings.append(f"Bootstrap 异常: {e}")
                logger.exception("Bootstrap 异常")
            finally:
                # 只有索引确实存在时才标记 initialized，否则后续可通过
                # _ensure_initialized 补救
                if await index_exists(root_path=self.project_path):
                    self._initialized = True
                # event 无论如何都 set，避免 pce_query/pce_impact 永远挂住
                self._bootstrap_event.set()
                if self._bootstrap_warnings:
                    logger.warning(
                        "Bootstrap 完成（有 %d 条 warning）", len(self._bootstrap_warnings)
                    )
                else:
                    logger.info("Bootstrap 完成（无 warning）")

    async def _ensure_initialized(self) -> None:
        """按需执行索引初始化（保留给 pce_sync 重连后使用）。"""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self._sync_lock:
                await self._run_index_refresh()
                self._initialized = True

    @staticmethod
    def _format_dirty_context(dirty: DirtyState) -> str:
        """将暂存区脏文件信息格式化为上下文注入文本。"""
        lines = ["[系统提示] 以下文件在上次索引后发生了变更，Memory 中对应信息可能不准确:"]
        for path in dirty.changed:
            lines.append(f"  - 变更: {path}")
        for path in dirty.deleted:
            lines.append(f"  - 删除: {path}")
        lines.append(
            "请使用 Serena 工具读取这些文件的最新内容进行验证，"
            "完成理解后调用 acknowledge_changes 确认认知。"
        )
        return "\n".join(lines)

    async def handle_query(
        self, query: str, session_id: str | None
    ) -> dict[str, Any]:
        """处理 pce_query 请求。"""
        await self._bootstrap_event.wait()
        # bootstrap 可能失败（索引未建成），此时 _ensure_initialized 会补救
        await self._ensure_initialized()

        # 注入暂存区脏文件信息到查询上下文
        dirty = await self.staging.list_unacknowledged()
        enriched_query = query
        if not dirty.empty:
            dirty_info = self._format_dirty_context(dirty)
            enriched_query = f"{query}\n\n{dirty_info}"

        response = await self.agent.query(
            question=enriched_query,
            session_id=session_id,
            memory_root=self.project_path,
            serena_client=self.serena_client,
            acknowledge_cb=self.staging.acknowledge,
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
        await self._bootstrap_event.wait()
        await self._ensure_initialized()
        # 将 file 参数附加到 target 描述中
        effective_target = f"{target} (file={file})" if file else target

        response = await self.agent.impact(
            target=effective_target,
            change_type=change_type,
            session_id=session_id,
            memory_root=self.project_path,
            serena_client=self.serena_client,
            acknowledge_cb=self.staging.acknowledge,
        )
        return response.model_dump(mode="json")

    async def handle_status(self, project_path: str | None) -> dict[str, Any]:
        """处理 pce_status 请求。"""
        root = Path(project_path).resolve() if project_path else self.project_path
        status = await get_status(root_path=root)
        staging_summary = await self.staging.summary()
        # 使用与当前项目对应的 InsightCache；若查询路径不同则临时构建
        cache = self.insight_cache if root == self.project_path else InsightCache(root)
        insight_stats = await cache.stats()
        return {
            **status,
            "initialized": await index_exists(root_path=root),
            "bootstrapping": not self._bootstrap_event.is_set(),
            "bootstrap_warnings": list(self._bootstrap_warnings),
            "project_path": str(root),
            "staging": staging_summary,
            "watcher_running": self.watcher.running,
            "insight_stats": insight_stats.model_dump(mode="json"),
        }

    async def handle_edit(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """处理写工具请求，直接透传给 Serena（不经过 PCE Agent 推理）。

        写操作成功后，自动将受影响文件记录到暂存区。
        暂存区条目留待下次会话开始时通过增量索引沉淀到 Memory。
        """
        if tool_name not in EDIT_TOOL_NAMES:
            raise SerenaClientError(f"不允许的写工具: {tool_name}")
        result = await self.serena_client.call(tool_name, arguments)

        # 从参数中提取受影响的文件路径，记录到暂存区
        affected = _extract_affected_path(tool_name, arguments)
        if affected:
            await self.staging.record_change(affected)

        return {"success": True, "tool": tool_name, "result": result}

    async def handle_sync(self) -> dict[str, Any]:
        """触发 Serena 断开重连并按需更新 PCE 索引。

        pce_sync 是上层 Agent 的显式同步信号，含义是"我做了大量修改，
        请立即将 Memory 更新到最新"。因此使用 list_pending_reindex
        获取全部待索引条目，索引完成后清空暂存区。
        """
        async with self._sync_lock:
            await self.serena_client.disconnect()
            await self.serena_client.connect(self.project_path, self.serena_path)

            dirty = await self.staging.list_pending_reindex()
            if not dirty.empty:
                all_paths = dirty.changed + dirty.deleted
                # 索引前快照 hash，防止索引期间新变更被提前确认
                hash_snapshot = await self.staging.snapshot_hashes(all_paths)
                logger.info(
                    f"pce_sync: 增量更新 {len(dirty.changed)} 变更, "
                    f"{len(dirty.deleted)} 删除"
                )
                snapshot = await build_index_incremental(
                    project_path=self.project_path,
                    serena_client=self.serena_client,
                    memory_root=self.project_path,
                    changed_files=dirty.changed,
                    deleted_files=dirty.deleted,
                )
                await self.staging.acknowledge_after_reindex(
                    all_paths, expected_hashes=hash_snapshot,
                )
                message = "Serena 已重连，PCE 索引增量更新完成"
            else:
                logger.info("pce_sync: 暂存区无变更，执行全量重建")
                snapshot = await build_index(
                    project_path=self.project_path,
                    serena_client=self.serena_client,
                    memory_root=self.project_path,
                )
                message = "Serena 已重连，PCE 索引全量重建完成"

            self._initialized = True
            logger.info(
                f"pce_sync: 完成 — "
                f"{snapshot.build_stats.total_files} 文件, "
                f"{snapshot.build_stats.total_symbols} 符号"
            )

        # Insight Cache stale sweep 在锁外执行，避免遍历 I/O 延长临界区
        stale_marked = await self.insight_cache.sweep_stale()
        logger.info(f"pce_sync: Insight Cache stale sweep 完成，新标记 {stale_marked} 条")

        return {
            "success": True,
            "message": message,
            "stats": snapshot.build_stats.model_dump(mode="json"),
        }


# ============================================================================
# 启动辅助
# ============================================================================


async def _ensure_serena(serena_path: Path) -> None:
    """确保 Serena 代码库已就绪,不存在时自动从 GitHub clone。

    前置要求: git 已安装并在 PATH 中。
    uv 会在首次 `uv run` 时自动安装 Serena 的 Python 依赖,无需手动操作。
    """
    if serena_path.is_dir():
        return

    logger.info(f"Serena 未安装,正在自动 clone 到 {serena_path} (需要 git 和网络连接)...")
    process = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth=1",
        "https://github.com/oraios/serena.git",
        str(serena_path),
    )
    returncode = await process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Serena 自动安装失败(返回码 {returncode})。"
            "请检查: 1) git 是否已安装 2) 网络连接是否正常 "
            "3) 或手动指定已安装的 Serena 路径: SERENA_PATH=/path/to/serena"
        )
    logger.info("Serena clone 完成")


# ============================================================================
# 主服务函数
# ============================================================================


async def serve() -> None:
    """启动 PCE MCP Server (stdio 模式)。

    读取环境变量:
        PCE_PROJECT_PATH: 目标项目根路径(默认当前目录)
        SERENA_PATH:      Serena 安装路径(默认 <PCE根目录>/serena,不存在时自动 clone)
        PCE_LOG_LEVEL:    日志级别(默认 INFO)
    """
    # 加载 PCE 根目录的 .env(不覆盖已有的环境变量,由调用方的 env/export 优先)
    pce_root = Path(__file__).parent.parent
    load_dotenv(dotenv_path=pce_root / ".env")

    # 配置日志
    log_level = os.getenv("PCE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    project_path = Path(os.getenv("PCE_PROJECT_PATH", str(Path.cwd()))).resolve()
    # 默认 Serena 路径为 PCE 包根目录下的 serena/,首次运行时自动 clone
    default_serena = str(pce_root / "serena")
    serena_path = Path(os.getenv("SERENA_PATH", default_serena)).resolve()

    logger.info(f"PCE Server 启动: project={project_path}, serena={serena_path}")

    # 确保 Serena 已安装(不存在时自动 clone；失败时记录日志并继续，首次工具调用时会报错)
    try:
        await _ensure_serena(serena_path)
    except Exception as e:
        logger.error(f"Serena 自动安装失败: {e}")

    # 初始化组件
    serena_timeout = _get_serena_timeout()
    serena_client = SerenaClient(timeout_seconds=serena_timeout)
    try:
        await serena_client.connect(project_path, serena_path)
    except Exception as e:
        logger.error(f"Serena 连接失败: {e},将在首次工具调用时重试")

    insight_cache = InsightCache(project_path)
    agent = PCEAgent(insight_cache=insight_cache)
    ctx = PCEContext(project_path, serena_path, serena_client, agent, insight_cache)

    # 启动文件监听
    await ctx.watcher.start()

    # eager bootstrap：启动后立即后台校验 Serena 项目激活并构建索引
    # pce_query / pce_impact 会 await _bootstrap_event.wait() 等待完成
    # pce_status 不阻塞，可随时查询 bootstrapping 状态和 warnings
    bootstrap_task = asyncio.create_task(ctx._bootstrap())

    # 创建并配置 MCP Server
    server = Server("pce")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        # 动态构建：写工具 schema 在 Serena 连接成功后才可用
        return _build_tools(ctx.serena_client.edit_tools_schema)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            await ctx.ensure_connected()

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

            if name == "pce_sync":
                result = await ctx.handle_sync()
                return _text_response(result)

            # 写工具透传：pce_{serena_tool_name} → serena_client.call(serena_tool_name, args)
            if name.startswith("pce_"):
                serena_tool = name[len("pce_"):]
                if serena_tool in EDIT_TOOL_NAMES:
                    result = await ctx.handle_edit(serena_tool, arguments)
                    return _text_response(result)

            return _error_response("UNKNOWN_TOOL", f"未知工具: {name}")

        except SerenaClientError as e:
            logger.error(f"Serena 客户端错误: {e}")
            return _error_response("SERENA_ERROR", str(e))
        except Exception as e:
            logger.exception(f"工具调用异常: {name}")
            return _error_response("INTERNAL_ERROR", f"内部错误: {e}")

    # 运行 stdio Server
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        bootstrap_task.cancel()
        try:
            await bootstrap_task
        except asyncio.CancelledError:
            pass
        await ctx.watcher.stop()


def main() -> None:
    """CLI 入口点。"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
