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
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .agent import PCEAgent
from .insight_cache import InsightCache
from .indexer import build_index, build_index_incremental
from .models import InitResponse
from .staging import DirtyState, FileWatcher, StagingArea
from .memory import get_status, index_exists, load_index
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


def _build_tools(
    *,
    edit_tools_schema: list[dict[str, Any]] | None = None,
) -> list[Tool]:
    """构建 MCP 工具定义。

    始终暴露核心工具（含 pce_query / pce_impact / pce_sync）；
    写工具是否暴露取决于 edit_tools_schema（需初始化后才有）。
    未初始化时调用核心工具，call_tool 会返回 NOT_INITIALIZED 错误引导 agent 先调 pce_init。
    """
    # 始终可用：init 与 status 不依赖项目初始化
    always_available: list[Tool] = [
        Tool(
            name="pce_init",
            description=_make_tool_description(
                summary="初始化 PCE 与目标项目的绑定，启动 Serena 并构建代码索引。",
                trigger=(
                    "会话开始后首次使用前调用一次。"
                    "调用成功前，pce_query / pce_impact / pce_sync 及写工具均不可用。"
                ),
                replaces=(
                    "替代通过环境变量传入项目路径的启动方式；"
                    "将项目绑定延迟到会话运行时显式完成，支持全局安装、零配置部署。"
                ),
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "目标项目根路径（绝对路径）",
                    },
                },
                "required": ["project_path"],
            },
        ),
        Tool(
            name="pce_status",
            description=_make_tool_description(
                summary="返回当前项目的 PCE 状态：初始化阶段、索引统计、暂存区信息。",
                trigger="诊断或调试时调用；也可在 pce_init 前后确认服务状态。",
                replaces="替代手动检查 .pce/ 目录文件与日志。",
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    # 始终暴露的核心查询/分析工具；未初始化时由 call_tool 返回 NOT_INITIALIZED 引导先调 pce_init
    post_init: list[Tool] = [
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
            name="pce_sync",
            description=_make_tool_description(
                summary="通知 PCE 代码库已发生变更，触发 Serena 重连并重建 PCE 索引。后续 pce_query/pce_impact 将看到最新代码。",
                trigger="上层 Agent 完成一批代码修改后调用，确保 PCE 索引与代码库同步。",
                replaces="替代手动重启 PCE 服务或等待索引自动失效。",
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    edit_tools = _build_edit_tools(edit_tools_schema or [])
    return [*always_available, *post_init, *edit_tools]


# ============================================================================
# 请求处理
# ============================================================================


class PCEContext:
    """持有 Server 的全局上下文,供工具处理器使用。"""

    def __init__(self) -> None:
        # 项目绑定状态
        self._bound_path: Path | None = None
        self._init_state: Literal[
            "uninitialized", "initializing", "initialized", "failed"
        ] = "uninitialized"
        self._last_init_error: str | None = None

        # 运行时组件（pce_init 后才创建）
        self.serena_client: SerenaClient | None = None
        self.agent: PCEAgent | None = None
        self.insight_cache: InsightCache | None = None
        self.staging: StagingArea | None = None
        self.watcher: FileWatcher | None = None

        # 锁与 bootstrap 信号
        self._sync_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._bootstrap_event = asyncio.Event()
        self._bootstrap_warnings: list[str] = []

    @property
    def project_path(self) -> Path:
        """返回已绑定的项目路径；未初始化时抛出异常。"""
        if self._bound_path is None:
            raise SerenaClientError("not initialized, call pce_init first")
        return self._bound_path

    def _require_initialized(self) -> None:
        """断言服务已完成初始化；否则抛出统一的错误。"""
        if self._init_state != "initialized":
            raise SerenaClientError("not initialized, call pce_init first")

    async def ensure_connected(self) -> None:
        """确保 Serena 连接已建立（断线后自动重连）。"""
        self._require_initialized()
        assert self.serena_client is not None
        if not self.serena_client.connected:
            logger.info("重新连接 Serena...")
            await self.serena_client.connect(self.project_path)

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

    async def _bootstrap(
        self,
        project_path: Path,
        init_mode: Literal["full_build", "retry_after_failure"],
    ) -> InitResponse:
        """执行初始化：创建运行时对象 → 连接 Serena → 激活项目 → 构建索引。

        同步阻塞到完成或失败。_bootstrap_event 由 handle_init 在状态切换时 clear，
        此处只负责在结束时 set（成功或失败都要 set，避免并发等待方永久挂住）。
        """
        warnings: list[str] = []
        self._bootstrap_warnings = []
        self._last_init_error = None
        # 注意：不在此处 clear event，handle_init 在切换到 initializing 时已 clear。

        # FileWatcher 在路径校验通过后立即启动，避免初始化期间遗漏文件变更
        if self.staging is None:
            self.staging = StagingArea(project_path)
        if self.watcher is None:
            self.watcher = FileWatcher(self.staging)
        if not self.watcher.running:
            await self.watcher.start()

        try:
            serena_timeout = _get_serena_timeout()
            self.serena_client = SerenaClient(timeout_seconds=serena_timeout)
            self.insight_cache = InsightCache(project_path)
            self.agent = PCEAgent(insight_cache=self.insight_cache)

            async with self._sync_lock:
                await self.serena_client.connect(project_path)

                # 显式校验 Serena 项目激活：启动时激活失败会被静默吞掉，
                # 再调一次可捕获失败并记录 warning，而不是直接报错
                try:
                    result = await self.serena_client.call(
                        "activate_project",
                        {"project": str(project_path)},
                    )
                    logger.info("Serena 项目激活成功: %s", result)
                except Exception as e:
                    warning = (
                        f"Serena 项目激活失败: {e}。"
                        "符号级索引可能不完整，请检查 Serena 项目配置。"
                    )
                    warnings.append(warning)
                    logger.warning(warning)

                await self._run_index_refresh()

            self._init_state = "initialized"
            self._bootstrap_warnings = list(warnings)
            self._bootstrap_event.set()
            logger.info(
                "Bootstrap 完成%s",
                f"（{len(warnings)} 条 warning）" if warnings else "（无 warning）",
            )

            snapshot = await load_index(root_path=project_path)
            file_count = snapshot.project_meta.file_count if snapshot else 0
            return InitResponse(
                initialized=True,
                status="initialized",
                project_path=str(project_path),
                project_name=project_path.name,
                file_count=file_count,
                init_mode=init_mode,
                warnings=self._bootstrap_warnings,
            )

        except Exception as e:
            self._init_state = "failed"
            self._last_init_error = str(e)
            self._bootstrap_warnings = list(warnings)
            self._bootstrap_event.set()
            logger.exception("PCE 初始化失败")
            # 清理半初始化的组件，重置为 None 以便下次重试时重新创建
            self.agent = None
            self.insight_cache = None
            if self.serena_client is not None:
                try:
                    await self.serena_client.disconnect()
                except Exception:
                    logger.exception("初始化失败后 Serena 清理异常")
                self.serena_client = None
            return InitResponse(
                initialized=False,
                status="init_failed",
                project_path=str(project_path),
                project_name=project_path.name,
                file_count=0,
                init_mode=init_mode,
                warnings=self._bootstrap_warnings,
                error=str(e),
            )

    async def handle_init(self, project_path_str: str) -> dict[str, Any]:
        """处理 pce_init 请求，同步阻塞到初始化完成。

        状态转换：
          uninitialized / failed → initializing → initialized | failed
          initialized（同路径）→ 直接返回（幂等）
          initialized / initializing（不同路径）→ 报错，需重启
        """
        if not project_path_str.strip():
            raise SerenaClientError("project_path 参数不能为空")

        resolved = Path(project_path_str).expanduser().resolve()
        if not resolved.exists():
            raise SerenaClientError(f"project path does not exist: {resolved}")
        if not resolved.is_dir():
            raise SerenaClientError(f"project path is not a directory: {resolved}")

        wait_for_existing = False
        init_mode: Literal["full_build", "retry_after_failure"] = "full_build"

        async with self._init_lock:
            # 路径冲突检查：只允许同路径操作
            if self._bound_path is not None and self._bound_path != resolved:
                raise SerenaClientError(
                    f"already bound to {self._bound_path}, restart to switch"
                )

            if self._init_state == "initialized":
                # 已完成：幂等快速返回
                snapshot = await load_index(root_path=resolved)
                return InitResponse(
                    initialized=True,
                    status="already_initialized",
                    project_path=str(resolved),
                    project_name=resolved.name,
                    file_count=snapshot.project_meta.file_count if snapshot else 0,
                    init_mode="reused",
                    warnings=list(self._bootstrap_warnings),
                ).model_dump(mode="json")

            elif self._init_state == "initializing":
                # 并发 init：等待正在进行的 bootstrap 完成
                wait_for_existing = True

            else:
                # "uninitialized" 或 "failed"：执行初始化
                init_mode = (
                    "retry_after_failure" if self._init_state == "failed" else "full_build"
                )
                self._bound_path = resolved
                self._init_state = "initializing"
                self._bootstrap_event.clear()

        if wait_for_existing:
            # 等待并发 init 结束后读取结果
            await self._bootstrap_event.wait()
            snapshot = await load_index(root_path=resolved)
            if self._init_state == "initialized":
                return InitResponse(
                    initialized=True,
                    status="already_initialized",
                    project_path=str(resolved),
                    project_name=resolved.name,
                    file_count=snapshot.project_meta.file_count if snapshot else 0,
                    init_mode="reused",
                    warnings=list(self._bootstrap_warnings),
                ).model_dump(mode="json")
            else:
                return InitResponse(
                    initialized=False,
                    status="init_failed",
                    project_path=str(resolved),
                    project_name=resolved.name,
                    file_count=0,
                    init_mode="retry_after_failure",
                    warnings=list(self._bootstrap_warnings),
                    error=self._last_init_error,
                ).model_dump(mode="json")

        response = await self._bootstrap(resolved, init_mode)
        return response.model_dump(mode="json")

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
        self._require_initialized()
        assert self.staging is not None
        assert self.agent is not None
        assert self.serena_client is not None

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
        self._require_initialized()
        assert self.agent is not None
        assert self.serena_client is not None
        assert self.staging is not None

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

    async def handle_status(self) -> dict[str, Any]:
        """处理 pce_status 请求（无需初始化，随时可调用）。"""
        root = self._bound_path
        status = (
            await get_status(root_path=root)
            if root is not None
            else {"last_index_time": None, "index_version": None, "memory_items_count": 0}
        )
        staging_summary = (
            await self.staging.summary()
            if self.staging is not None
            else {
                "pending_reindex": 0,
                "pending_changed": 0,
                "pending_deleted": 0,
                "session_acknowledged": 0,
            }
        )
        insight_stats = (
            await self.insight_cache.stats() if self.insight_cache is not None else None
        )
        return {
            **status,
            "initialized": self._init_state == "initialized",
            "init_state": self._init_state,
            "bootstrapping": self._init_state == "initializing",
            "last_init_error": self._last_init_error,
            "bootstrap_warnings": list(self._bootstrap_warnings),
            "project_path": str(root) if root is not None else None,
            "staging": staging_summary,
            "watcher_running": self.watcher.running if self.watcher is not None else False,
            "insight_stats": (
                insight_stats.model_dump(mode="json") if insight_stats is not None else None
            ),
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
        self._require_initialized()
        assert self.serena_client is not None
        assert self.staging is not None

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
        self._require_initialized()
        assert self.serena_client is not None
        assert self.staging is not None
        assert self.insight_cache is not None

        async with self._sync_lock:
            await self.serena_client.disconnect()
            await self.serena_client.connect(self.project_path)

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

            self._init_state = "initialized"
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


# ============================================================================
# 主服务函数
# ============================================================================


async def serve() -> None:
    """启动 PCE MCP Server (stdio 模式)。

    读取环境变量:
        PCE_LOG_LEVEL:      日志级别（默认 INFO）
        PCE_SERENA_TIMEOUT: Serena MCP 超时秒数（默认 180）

    项目路径由 agent 在会话开始时通过 pce_init 工具传入，无需环境变量配置。
    """
    # 加载 PCE 根目录的 .env（不覆盖已有环境变量，调用方的 env/export 优先）
    pce_root = Path(__file__).parent.parent
    load_dotenv(dotenv_path=pce_root / ".env")

    log_level = os.getenv("PCE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger.info("PCE Server 启动（等待 pce_init 绑定项目）")
    ctx = PCEContext()

    server = Server(
        "pce",
        instructions=(
            "PCE 工作流: 第一步必须调用 pce_init 绑定并初始化项目。\n"
            "初始化完成后,再使用 pce_query、pce_impact、pce_sync 及相关写工具。\n"
            "若未初始化就调用其他工具,会返回 NOT_INITIALIZED 错误。"
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        edit_schema = (
            ctx.serena_client.edit_tools_schema
            if ctx.serena_client is not None
            else None
        )
        return _build_tools(edit_tools_schema=edit_schema)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            # pce_init 和 pce_status 不依赖项目初始化，优先分流
            if name == "pce_init":
                project_path = arguments.get("project_path", "")
                result = await ctx.handle_init(project_path)
                return _text_response(result)

            if name == "pce_status":
                result = await ctx.handle_status()
                return _text_response(result)

            # 其余工具在未初始化时统一返回错误
            if ctx._init_state != "initialized":
                return _error_response(
                    "NOT_INITIALIZED",
                    "not initialized, call pce_init first",
                )

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

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        # 各自独立清理，互不干扰
        if ctx.watcher is not None:
            try:
                await ctx.watcher.stop()
            except Exception:
                logger.exception("watcher 停止异常")
        if ctx.serena_client is not None:
            try:
                await ctx.serena_client.disconnect()
            except Exception:
                logger.exception("serena_client 断开异常")


def main() -> None:
    """CLI 入口点。"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
