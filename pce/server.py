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
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from .agent import PCEAgent
from .baseline_maintenance import seed_initial_file_baselines_if_missing
from .digest_agent import run_digest, should_run_digest
from .file_discovery import (
    HARD_SKIP_DIRS,
    filter_visible_paths,
    should_track_existing_file,
    should_track_deleted_path,
)
from .insight_cache import InsightCache
from .indexer import build_index, build_index_incremental
from .models import InitResponse, LanguageHealthReport
from .serena_language_health import (
    preflight_serena_language_health,
    verify_serena_language_health,
)
from .staging import DirtyState, FileWatcher, StagingArea
from .memory import get_status, index_exists, load_file_baseline, load_index
from ._env import configure_litellm_runtime
from pce_v2 import ImpactRequest as V2ImpactRequest
from pce_v2 import PCEngine as PCEngineV2
from pce_v2 import QueryRequest as V2QueryRequest
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


def _use_v2_query_impact() -> bool:
    raw = os.getenv("PCE_USE_V2_QUERY_IMPACT", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _make_tool_description(
    *,
    purpose: str,
    use_when: str,
    best_practice: str,
    avoid_when: str | None = None,
) -> str:
    """Assemble agent-facing tool description.

    (原中文字段: 用途 / 适用时机 / 最佳实践 / 避免误用)
    """
    parts = [
        f"Purpose: {purpose}",
        f"When to use: {use_when}",
        f"Best practice: {best_practice}",
    ]
    if avoid_when:
        parts.append(f"Avoid: {avoid_when}")
    return "\n".join(parts)


def _text_response(content: Any) -> list[TextContent]:
    """将任意数据序列化为 MCP TextContent 列表。"""
    return [TextContent(type="text", text=json.dumps(content, ensure_ascii=False, indent=2))]


async def _hash_existing_file(path: Path) -> str | None:
    def _hash() -> str | None:
        if not path.exists() or not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    return await asyncio.to_thread(_hash)


async def _scan_trackable_files(project_root: Path) -> list[str]:
    def _walk() -> list[str]:
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(project_root):
            dirnames[:] = [name for name in dirnames if name not in HARD_SKIP_DIRS]
            base = Path(dirpath)
            for filename in filenames:
                rel_path = (base / filename).relative_to(project_root).as_posix()
                if should_track_existing_file(project_root, rel_path):
                    results.append(rel_path)
        return sorted(set(results))

    return await asyncio.to_thread(_walk)


async def _collect_baseline_paths(project_root: Path) -> set[str]:
    def _collect() -> set[str]:
        base_dir = project_root / ".pce" / "baselines" / "files"
        if not base_dir.exists():
            return set()
        results: set[str] = set()
        for path in base_dir.rglob("*.json"):
            rel = path.relative_to(base_dir).as_posix()
            if rel.endswith(".json"):
                results.add(rel[:-5])
        return results

    return await asyncio.to_thread(_collect)


def _markdown_response(content: str) -> list[TextContent]:
    """返回 Markdown 文本响应。"""
    return [TextContent(type="text", text=content)]


def _error_response(error_code: str, message: str) -> list[TextContent]:
    """统一错误响应格式。"""
    return _text_response(
        {
            "success": False,
            "error_code": error_code,
            "error_message": message,
        }
    )


def _structured_tool_result(content: dict[str, Any], *, is_error: bool = False) -> CallToolResult:
    """返回带 structuredContent 的 MCP 工具结果。"""
    return CallToolResult(
        content=_text_response(content),
        structuredContent=content,
        isError=is_error,
    )


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


def _format_exception_brief(exc: BaseException) -> str:
    """将异常格式化为稳定、非空的 warning 文本。"""
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__}: {exc!r}"


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
        # pce_init 原中文描述:
        #   用途: 绑定目标项目并初始化 PCE 运行时，为后续 query / impact / sync 建立索引与导航上下文。
        #   适用时机: 进入一个新项目或会话首次使用 PCE 时调用。在它成功之前，不应调用 pce_query、pce_impact、pce_sync 或写工具。
        #   最佳实践: 每个会话通常只需调用一次并等待成功。只有在需要切换项目或初始化失败后重试时，才再次调用。
        #   避免误用: 不要把它当作代码查询工具使用；它负责建立上下文，不直接回答代码问题。
        Tool(
            name="pce_init",
            description=_make_tool_description(
                purpose=(
                    "Bind a target project and initialize the PCE runtime, "
                    "building the index and navigation context required by query / impact / sync."
                ),
                use_when=(
                    "Call this when entering a new project or the first time PCE is needed in a session. "
                    "Do NOT call pce_query, pce_impact, pce_sync, or edit tools before pce_init succeeds."
                ),
                best_practice=(
                    "Typically called once per session and awaited until success. "
                    "Only call again when switching projects or retrying after a failure."
                ),
                avoid_when=(
                    "Do not use this as a code query tool; "
                    "it establishes context but does not directly answer code questions."
                ),
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path to the target project root.",
                    },
                },
                "required": ["project_path"],
            },
        ),
        # pce_status 原中文描述:
        #   用途: 返回当前服务与项目状态，例如初始化阶段、索引统计、暂存区与 warning。
        #   适用时机: 需要确认 PCE 是否可用、索引是否已建立，或排查 init / query / impact / sync 异常时使用。
        #   最佳实践: 把它当作诊断工具使用。当不确定是否应先 init 或 sync 时，先查看 status。
        #   避免误用: 不要把它当作代码理解工具使用；它不负责定位入口、调用链或影响边界。
        Tool(
            name="pce_status",
            description=_make_tool_description(
                purpose=(
                    "Return current service and project status, "
                    "including initialization phase, index statistics, staging area, and warnings."
                ),
                use_when=(
                    "Use when you need to confirm whether PCE is available, "
                    "whether the index has been built, or to diagnose init / query / impact / sync issues."
                ),
                best_practice=(
                    "Treat this as a diagnostic tool. "
                    "When unsure whether to call init or sync first, check status."
                ),
                avoid_when=(
                    "Do not use this as a code understanding tool; "
                    "it does not locate entry points, call chains, or impact boundaries."
                ),
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

    # 始终暴露的核心查询/分析工具；未初始化时由 call_tool 返回 NOT_INITIALIZED 引导先调 pce_init
    post_init: list[Tool] = [
        # pce_query 原中文描述:
        #   用途: 用于代码库定位与理解，是高层代码搜索与建立任务认知的首选工具。
        #         适合入口、主调用链、模块职责、候选文件范围和项目/模块整体理解。
        #   适用时机: 当目标位置尚不明确时使用。当你不知道信息在哪个文件、需要先理解某个能力大致如何实现、
        #             想找入口/主调用链/模块职责、或需要先缩小搜索范围时，应优先使用。不要先手工遍历目录或批量读文件。
        #   最佳实践: 优先用自然语言描述你要理解的问题，而不是只给精确标识符。
        #             可直接要求返回 file:line、name_path、候选文件列表、调用链摘要或按模块归纳的结果。
        #   避免误用: 当你已经知道精确文件或精确标识符，只需要查看局部实现或做精确字符串匹配时，不必先调用 pce_query。
        #             若 target 已明确且任务变成"改它会影响哪里"，应转用 pce_impact。
        Tool(
            name="pce_query",
            description=_make_tool_description(
                purpose=(
                    "The primary tool for codebase navigation and understanding. "
                    "Best suited for project-level understanding, architecture overviews, major module responsibilities, "
                    "entry points, main call chains, candidate file scopes, and overall project/module comprehension."
                ),
                use_when=(
                    "Use when the target location is unclear. "
                    "When you do not know which file contains the information, "
                    "need to understand what the project or a subsystem does, "
                    "need a rough architecture or module-level picture, "
                    "want to find entry points / main call chains / module responsibilities, "
                    "or need to narrow down the search scope — use this tool first. "
                    "Do NOT manually traverse directories, broadly inspect the repo, or batch-read files before trying pce_query."
                ),
                best_practice=(
                    "Prefer describing your question in natural language rather than just giving exact identifiers. "
                    "This tool is especially appropriate for questions like what the project does, how a subsystem is organized, where a feature lives, or which modules participate in a workflow. "
                    "You can request file:line references, name_path, candidate file lists, call chain summaries, or results grouped by module."
                ),
                avoid_when=(
                    "When you already know the exact file or exact identifier and only need to view "
                    "local implementation or do exact string matching, pce_query is not necessary. "
                    "If the target is already clear and the task becomes 'what will be affected by changing it', "
                    "switch to pce_impact instead."
                ),
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question, e.g. 'Where is the entry point for authentication logic?'",
                    },
                },
                "required": ["query"],
            },
        ),
        # pce_impact 原中文描述:
        #   用途: 用于分析已知变更目标的影响边界，是改动前理解波及面的首选工具。
        #         重点输出直接调用点、直接消费者、主要传播链、风险与建议修改顺序。
        #   适用时机: 当你已经明确要改哪个符号、字段、接口契约或文件，并想先了解它会影响哪里时，优先使用。
        #   最佳实践: 尽量提供明确的 target；若已知符号所在文件，也一起提供 file 以加速定位。
        #             优先把问题提成具体变更，例如"修改某字段""调整某函数签名""删除某文件后会影响哪里"。
        #   避免误用: 如果 target 仍然模糊、还在多个候选之间摇摆，不要用 impact 代替定位步骤；应先使用 pce_query 收敛目标。
        #             若只是想看局部实现或精确定义，也不必先调用 impact。
        Tool(
            name="pce_impact",
            description=_make_tool_description(
                purpose=(
                    "The primary tool for analyzing the impact boundary of a known change target. "
                    "Outputs direct call sites, direct consumers, main propagation chains, "
                    "risks, and suggested modification order."
                ),
                use_when=(
                    "Use when you already know which symbol, field, interface contract, or file to change, "
                    "and want to understand what will be affected before making the change."
                ),
                best_practice=(
                    "Provide an explicit target; if you know the file containing the symbol, "
                    "also provide the file parameter to speed up resolution. "
                    "Frame the question as a concrete change, e.g. 'modify field X', "
                    "'change function signature of Y', 'what breaks if file Z is deleted'."
                ),
                avoid_when=(
                    "If the target is still ambiguous or you are choosing between multiple candidates, "
                    "do not use impact as a substitute for the discovery step — use pce_query first to converge on the target. "
                    "If you only want to view local implementation or exact definitions, impact is not necessary."
                ),
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The change target — a symbol name (e.g. UserSession) or file path.",
                    },
                    "change_type": {
                        "type": "string",
                        "description": "Type of change: modify | rename | delete | add_field | change_signature",
                        "enum": ["modify", "rename", "delete", "add_field", "change_signature"],
                    },
                    "file": {
                        "type": "string",
                        "description": "File path containing the symbol (optional, speeds up resolution if provided).",
                    },
                },
                "required": ["target", "change_type"],
            },
        ),
        # pce_sync 原中文描述:
        #   用途: 在代码库修改后同步 Serena 与 PCE 的索引状态，使后续 query / impact 基于最新代码工作。
        #   适用时机: 完成一批代码修改、文件删除、重命名或结构调整后使用。
        #   最佳实践: 把它作为批量同步步骤使用；通常在完成一轮修改后再调用一次，而不是每改一小处就立刻同步。
        #   避免误用: 不要把它当作代码理解工具使用；它负责刷新状态，不负责解释代码。
        Tool(
            name="pce_sync",
            description=_make_tool_description(
                purpose=(
                    "Synchronize Serena and PCE index state after codebase modifications, "
                    "so subsequent query / impact calls work against the latest code."
                ),
                use_when=(
                    "Use after completing a batch of code modifications, file deletions, "
                    "renames, or structural changes."
                ),
                best_practice=(
                    "Use this as a batch synchronization step; "
                    "typically call once after completing a round of changes, "
                    "rather than after every small edit."
                ),
                avoid_when=(
                    "Do not use this as a code understanding tool; "
                    "it refreshes index state but does not explain code."
                ),
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
        self._init_state: Literal["uninitialized", "initializing", "initialized", "failed"] = (
            "uninitialized"
        )
        self._last_init_error: str | None = None

        # 运行时组件（pce_init 后才创建）
        self.agent: PCEAgent | None = None
        self.v2_engine = PCEngineV2()
        self.insight_cache: InsightCache | None = None
        self.staging: StagingArea | None = None
        self.watcher: FileWatcher | None = None
        # 只缓存纯数据（schema），不要跨请求持有活跃 SerenaClient。
        # mcp/anyio 的 session/context manager 绑定创建它的任务，请求结束后跨任务复用会触发
        # cancel scope 错位，导致 transport 被关闭。
        self._edit_tools_schema: list[dict[str, Any]] = []

        # 锁与 bootstrap 信号
        self._sync_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._bootstrap_event = asyncio.Event()
        self._bootstrap_warnings: list[str] = []
        self._language_health_report: LanguageHealthReport | None = None

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

    @asynccontextmanager
    async def serena_session(self) -> Any:
        """在当前请求任务内建立并关闭 Serena 连接，避免跨任务持有 anyio context。"""
        self._require_initialized()
        client = SerenaClient(timeout_seconds=_get_serena_timeout())
        try:
            await client.connect(self.project_path)
            yield client
        finally:
            try:
                await client.disconnect()
            except Exception:
                logger.exception("Serena 会话关闭异常")

    async def _run_index_refresh(self, serena_client: SerenaClient) -> dict[str, Any]:
        """执行索引刷新逻辑，返回统一的刷新结果。

        调用方负责加锁（_sync_lock）与状态位更新。
        1. 索引不存在 → 全量构建 + 清空暂存区
        2. 索引存在但暂存区有累积变更 → 增量更新 + 清空暂存区
        3. 索引存在且无变更 → 直接跳过，返回当前（空）DirtyState
        """
        assert self.staging is not None
        dirty = await self._backfill_dirty_if_needed()
        if not await index_exists(root_path=self.project_path):
            logger.info("PCE 索引不存在，开始全量构建...")
            all_paths = dirty.changed + dirty.deleted
            hash_snapshot = await self.staging.snapshot_hashes(all_paths) if all_paths else None
            snapshot = await build_index(
                project_path=self.project_path,
                serena_client=serena_client,
                memory_root=self.project_path,
            )
            if all_paths:
                await self.staging.acknowledge_after_reindex(
                    all_paths,
                    expected_hashes=hash_snapshot,
                )
            logger.info("PCE 索引全量构建完成")
            return {
                "dirty_state": dirty,
                "snapshot": snapshot,
                "refresh_mode": "full_rebuild",
                "message": "Serena 已重连，PCE 索引全量重建完成",
            }

        if not dirty.empty:
            all_paths = dirty.changed + dirty.deleted
            hash_snapshot = await self.staging.snapshot_hashes(all_paths)
            logger.info(
                f"检测到 {len(dirty.changed)} 个变更、"
                f"{len(dirty.deleted)} 个删除，执行增量索引更新"
            )
            snapshot = await build_index_incremental(
                project_path=self.project_path,
                serena_client=serena_client,
                memory_root=self.project_path,
                changed_files=dirty.changed,
                deleted_files=dirty.deleted,
            )
            await self.staging.acknowledge_after_reindex(all_paths, expected_hashes=hash_snapshot)
            logger.info("增量索引更新完成")
            return {
                "dirty_state": dirty,
                "snapshot": snapshot,
                "refresh_mode": "incremental",
                "message": "Serena 已重连，PCE 索引增量更新完成",
            }

        snapshot = await load_index(root_path=self.project_path)
        return {
            "dirty_state": dirty,
            "snapshot": snapshot,
            "refresh_mode": "noop",
            "message": "暂存区无变更，索引已是最新",
        }

    async def _backfill_dirty_if_needed(self) -> DirtyState:
        """当 watcher 离线导致暂存区为空时，基于 baseline/索引做轻量补录。"""
        assert self.staging is not None

        dirty = await self.staging.list_pending_reindex()
        if not dirty.empty:
            return dirty

        project_root = self.project_path
        snapshot = await load_index(root_path=project_root)
        baseline_paths = await _collect_baseline_paths(project_root)
        if not baseline_paths and snapshot is None:
            return dirty
        current_files = await _scan_trackable_files(project_root)
        current_set = set(current_files)

        expected_paths = set(baseline_paths)
        if not expected_paths and snapshot is not None:
            expected_paths = {str(entry.file_meta.path) for entry in snapshot.entries}

        created_paths = sorted(current_set - expected_paths)
        deleted_paths = sorted(path for path in (expected_paths - current_set) if should_track_deleted_path(path))

        modified_paths: list[str] = []
        if baseline_paths:
            for rel_path in sorted(current_set & baseline_paths):
                baseline = await load_file_baseline(rel_path, root_path=project_root)
                if baseline is None:
                    continue
                current_hash = await _hash_existing_file(project_root / rel_path)
                if current_hash is not None and current_hash != baseline.content_hash:
                    modified_paths.append(rel_path)

        if not created_paths and not deleted_paths and not modified_paths:
            return dirty

        logger.info(
            "离线 dirty 补录: created=%d modified=%d deleted=%d",
            len(created_paths),
            len(modified_paths),
            len(deleted_paths),
        )

        for rel_path in sorted(set([*created_paths, *modified_paths])):
            await self.staging.record_change(rel_path)
        for rel_path in deleted_paths:
            await self.staging.record_change(rel_path, deleted=True)

        return await self.staging.list_pending_reindex()

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
        self._language_health_report = None
        self._last_init_error = None
        bootstrap_start = time.monotonic()
        # 注意：不在此处 clear event，handle_init 在切换到 initializing 时已 clear。

        # FileWatcher 在路径校验通过后立即启动，避免初始化期间遗漏文件变更
        if self.staging is None:
            self.staging = StagingArea(project_path)
        if self.watcher is None:
            self.watcher = FileWatcher(self.staging)
        if not self.watcher.running:
            await self.watcher.start()

        # 加载项目级 .env（不覆盖已有环境变量）
        # 优先级：MCP config env > 系统环境变量 > 项目 .env
        load_dotenv(dotenv_path=project_path / ".env", override=False)

        try:
            serena_timeout = _get_serena_timeout()
            self.insight_cache = InsightCache(project_path)
            self.agent = PCEAgent(insight_cache=self.insight_cache)
            try:
                language_health = await preflight_serena_language_health(project_path)
            except Exception as exc:
                warning = f"语言健康预检失败（已降级）: {_format_exception_brief(exc)}"
                warnings.append(warning)
                logger.warning(warning)
            else:
                self._language_health_report = language_health
                warnings.extend(language_health.warnings)

            async with self._sync_lock:
                serena_connect_start = time.monotonic()
                async with SerenaClient.create(
                    project_path,
                    timeout_seconds=serena_timeout,
                ) as serena_client:
                    logger.info(
                        "Bootstrap 阶段耗时: serena_connect=%.2fs",
                        time.monotonic() - serena_connect_start,
                    )
                    self._edit_tools_schema = serena_client.edit_tools_schema
                    if self._language_health_report is not None:
                        try:
                            language_health = await verify_serena_language_health(
                                self._language_health_report,
                                serena_client,
                            )
                        except Exception as exc:
                            warning = f"语言健康校验失败（已降级）: {_format_exception_brief(exc)}"
                            warnings.append(warning)
                            logger.warning(warning)
                        else:
                            self._language_health_report = language_health
                            warnings = [*warnings, *[
                                item for item in language_health.warnings if item not in warnings
                            ]]

                    # 显式校验 Serena 项目激活：启动时激活失败会被静默吞掉，
                    # 再调一次可捕获失败并记录 warning，而不是直接报错
                    activate_start = time.monotonic()
                    try:
                        result = await serena_client.call(
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
                    finally:
                        logger.info(
                            "Bootstrap 阶段耗时: activate_project=%.2fs",
                            time.monotonic() - activate_start,
                        )

                    await self._run_post_index_pipeline(
                        serena_client=serena_client,
                        warnings=warnings,
                        phase_label="Bootstrap",
                    )

            self._init_state = "initialized"
            self._bootstrap_warnings = list(warnings)
            self._bootstrap_event.set()
            logger.info(
                "Bootstrap 完成%s",
                f"（{len(warnings)} 条 warning）" if warnings else "（无 warning）",
            )
            logger.info(
                "Bootstrap 总耗时: %.2fs",
                time.monotonic() - bootstrap_start,
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
                language_health_report=self._language_health_report,
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
            self._edit_tools_schema = []
            return InitResponse(
                initialized=False,
                status="init_failed",
                project_path=str(project_path),
                project_name=project_path.name,
                file_count=0,
                init_mode=init_mode,
                warnings=self._bootstrap_warnings,
                language_health_report=self._language_health_report,
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
                raise SerenaClientError(f"already bound to {self._bound_path}, restart to switch")

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
                    language_health_report=self._language_health_report,
                ).model_dump(mode="json")

            elif self._init_state == "initializing":
                # 并发 init：等待正在进行的 bootstrap 完成
                wait_for_existing = True

            else:
                # "uninitialized" 或 "failed"：执行初始化
                init_mode = "retry_after_failure" if self._init_state == "failed" else "full_build"
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
                    language_health_report=self._language_health_report,
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
                    language_health_report=self._language_health_report,
                    error=self._last_init_error,
                ).model_dump(mode="json")

        response = await self._bootstrap(resolved, init_mode)
        return response.model_dump(mode="json")

    _DIRTY_INJECT_MAX_FILES: int = 50  # 注入脏文件列表的上限

    def _filter_visible_dirty(self, dirty: DirtyState) -> DirtyState:
        """过滤掉不应暴露给 Agent 的 dirty 路径。"""
        return DirtyState(
            changed=filter_visible_paths(self.project_path, dirty.changed),
            deleted=filter_visible_paths(self.project_path, dirty.deleted),
        )

    @classmethod
    def _format_dirty_context(cls, dirty: DirtyState) -> str:
        """将暂存区脏文件信息格式化为上下文注入文本。

        超出上限时截断并提示 Agent 先调 pce_sync。
        """
        all_paths = dirty.changed + dirty.deleted
        total = len(all_paths)
        cap = cls._DIRTY_INJECT_MAX_FILES

        # 原中文提示文本:
        #   [系统提示] 以下文件在上次索引后发生了变更，Memory 中对应信息可能不准确:
        #   变更: / 删除:
        #   ... 以及另外 N 个变更文件未列出。建议先调用 pce_sync 更新索引后再继续查询。
        #   请使用 Serena 工具读取这些文件的最新内容进行验证，完成理解后调用 acknowledge_changes 确认认知。
        lines = ["[System Notice] The following files have changed since the last index — Memory entries for them may be stale:"]
        for path in dirty.changed[:cap]:
            lines.append(f"  - Changed: {path}")
        remaining = max(0, cap - len(dirty.changed))
        for path in dirty.deleted[:remaining]:
            lines.append(f"  - Deleted: {path}")

        if total > cap:
            lines.append(
                f"  ... and {total - cap} more changed files not listed. "
                "Consider calling pce_sync to update the index before continuing."
            )

        lines.append(
            "Use Serena tools to read the latest content of these files for verification, "
            "then call acknowledge_changes to confirm awareness."
        )
        return "\n".join(lines)

    async def handle_query(self, query: str) -> dict[str, Any]:
        """处理 pce_query 请求。"""
        self._require_initialized()
        assert self.staging is not None
        assert self.agent is not None

        # 注入暂存区脏文件信息到查询上下文
        dirty = await self.staging.list_unacknowledged()
        visible_dirty = self._filter_visible_dirty(dirty)
        enriched_query = query
        if not visible_dirty.empty:
            dirty_info = self._format_dirty_context(visible_dirty)
            enriched_query = f"{query}\n\n{dirty_info}"

        if _use_v2_query_impact():
            try:
                result = await self.v2_engine.run_query(
                    self.project_path,
                    V2QueryRequest(question=enriched_query),
                )
                result["engine"] = "v2"
                return result
            except Exception:
                logger.exception("v2 query 执行失败，回退旧版主线")

        async with self.serena_session() as serena_client:
            response = await self.agent.query(
                question=enriched_query,
                memory_root=self.project_path,
                serena_client=serena_client,
                acknowledge_cb=self.staging.acknowledge,
            )
        payload = response.model_dump(mode="json")
        payload["answer"] = response.answer
        payload["engine"] = "legacy"
        return payload

    async def handle_impact(
        self,
        target: str,
        change_type: str,
        file: str | None,
    ) -> dict[str, Any]:
        """处理 pce_impact 请求。"""
        self._require_initialized()
        assert self.agent is not None
        assert self.staging is not None

        effective_target = f"{target} (file={file})" if file else target

        if _use_v2_query_impact():
            try:
                result = await self.v2_engine.run_impact(
                    self.project_path,
                    V2ImpactRequest(target=target, change_type=change_type, file=file),
                )
                result["engine"] = "v2"
                return result
            except Exception:
                logger.exception("v2 impact 执行失败，回退旧版主线")

        async with self.serena_session() as serena_client:
            response = await self.agent.impact(
                target=effective_target,
                change_type=change_type,
                memory_root=self.project_path,
                serena_client=serena_client,
                acknowledge_cb=self.staging.acknowledge,
            )
        payload = response.model_dump(mode="json")
        payload["answer"] = response.answer
        payload["engine"] = "legacy"
        return payload

    async def handle_status(self) -> dict[str, Any]:
        """处理 pce_status 请求（无需初始化，随时可调用）。"""
        root = self._bound_path
        status = (
            await get_status(root_path=root)
            if root is not None
            else {"last_index_time": None, "index_version": None, "annotation_modules_count": 0}
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
        insight_stats = await self.insight_cache.stats() if self.insight_cache is not None else None
        return {
            **status,
            "initialized": self._init_state == "initialized",
            "init_state": self._init_state,
            "bootstrapping": self._init_state == "initializing",
            "last_init_error": self._last_init_error,
            "bootstrap_warnings": list(self._bootstrap_warnings),
            "language_health_report": (
                self._language_health_report.model_dump(mode="json")
                if self._language_health_report is not None
                else None
            ),
            "project_path": str(root) if root is not None else None,
            "staging": staging_summary,
            "watcher_running": self.watcher.running if self.watcher is not None else False,
            "insight_stats": (
                insight_stats.model_dump(mode="json") if insight_stats is not None else None
            ),
        }

    async def handle_edit(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """处理写工具请求，直接透传给 Serena（不经过 PCE Agent 推理）。

        写操作成功后，自动将受影响文件记录到暂存区。
        暂存区条目留待下次会话开始时通过增量索引沉淀到 Memory。
        """
        if tool_name not in EDIT_TOOL_NAMES:
            raise SerenaClientError(f"不允许的写工具: {tool_name}")
        self._require_initialized()
        assert self.staging is not None

        async with self.serena_session() as serena_client:
            result = await serena_client.call(tool_name, arguments)

        # 从参数中提取受影响的文件路径，记录到暂存区
        affected = _extract_affected_path(tool_name, arguments)
        if affected:
            await self.staging.record_change(affected)

        return {"success": True, "tool": tool_name, "result": result}

    async def _run_post_index_cognition_pipeline(
        self,
        *,
        serena_client: SerenaClient,
        dirty_state: DirtyState,
        warnings: list[str] | None,
        phase_label: str,
    ) -> dict[str, Any] | None:
        assert self.insight_cache is not None
        should_digest, digest_reason = await should_run_digest(
            project_root=self.project_path,
            insight_cache=self.insight_cache,
            dirty_state=dirty_state,
        )
        if not should_digest:
            logger.info("%s Digest 已跳过：%s", phase_label, digest_reason)
            return None

        digest_start = time.monotonic()
        try:
            digest_result = await run_digest(
                project_root=self.project_path,
                serena_client=serena_client,
                insight_cache=self.insight_cache,
                dirty_state=dirty_state,
                skip_initial_sweep=True,
            )
            if warnings is not None:
                for item in digest_result.get("warnings", []):
                    warnings.append(f"Digest: {item}")
            logger.info(
                "%s Digest 完成: resolved=%d pending=%d deleted_insights=%d",
                phase_label,
                digest_result.get("resolved_tasks", 0),
                digest_result.get("pending_tasks", 0),
                digest_result.get("deleted_insights", 0),
            )
            return digest_result
        except Exception as e:
            warning = f"{phase_label} Digest 失败（不影响主流程）: {_format_exception_brief(e)}"
            if warnings is not None:
                warnings.append(warning)
            logger.warning(warning)
            return None
        finally:
            logger.info(
                "%s 阶段耗时: digest=%.2fs",
                phase_label,
                time.monotonic() - digest_start,
            )

    async def _run_post_index_pipeline(
        self,
        *,
        serena_client: SerenaClient,
        warnings: list[str] | None,
        phase_label: str,
    ) -> dict[str, Any]:
        """统一的 index 后处理编排入口。

        顺序固定为：
        1. index refresh（内部已覆盖 navigation 增量更新/重建）
        2. seed baselines
        3. digest gate + digest
        """
        index_refresh_start = time.monotonic()
        refresh_result = await self._run_index_refresh(serena_client)
        dirty_state: DirtyState = refresh_result["dirty_state"]
        logger.info(
            "%s 阶段耗时: index_refresh=%.2fs (changed=%d deleted=%d mode=%s)",
            phase_label,
            time.monotonic() - index_refresh_start,
            len(dirty_state.changed),
            len(dirty_state.deleted),
            refresh_result["refresh_mode"],
        )

        baseline_seed_start = time.monotonic()
        await seed_initial_file_baselines_if_missing(project_root=self.project_path)
        logger.info(
            "%s 阶段耗时: seed_initial_baselines=%.2fs",
            phase_label,
            time.monotonic() - baseline_seed_start,
        )

        await self._run_post_index_cognition_pipeline(
            serena_client=serena_client,
            dirty_state=dirty_state,
            warnings=warnings,
            phase_label=phase_label,
        )
        return refresh_result

    async def handle_sync(self) -> dict[str, Any]:
        """触发 Serena 断开重连并按需更新 PCE 索引。

        pce_sync 是上层 Agent 的显式同步信号，含义是"我做了大量修改，
        请立即将 Memory 更新到最新"。因此使用 list_pending_reindex
        获取全部待索引条目，索引完成后清空暂存区。
        """
        self._require_initialized()
        assert self.staging is not None
        assert self.insight_cache is not None

        async with self._sync_lock:
            digest_warnings: list[str] = []

            async with self.serena_session() as serena_client:
                refresh_result = await self._run_post_index_pipeline(
                    serena_client=serena_client,
                    warnings=digest_warnings,
                    phase_label="pce_sync",
                )
                snapshot = refresh_result["snapshot"]
                message = refresh_result["message"]

            self._init_state = "initialized"
            logger.info(
                f"pce_sync: 完成 — "
                f"{snapshot.build_stats.total_files} 文件, "
                f"{snapshot.build_stats.total_symbols} 符号"
            )

        return {
            "success": True,
            "message": message,
            "stats": snapshot.build_stats.model_dump(mode="json"),
            "warnings": digest_warnings,
        }


# ============================================================================
# 启动辅助
# ============================================================================


# ============================================================================
# 主服务函数
# ============================================================================


async def serve() -> None:
    """启动 PCE MCP Server (stdio 模式)。

    读取环境变量：
        PCE_LOG_LEVEL:      日志级别（默认 INFO）
        PCE_SERENA_TIMEOUT: Serena MCP 超时秒数（默认 180）

    项目路径由 agent 在会话开始时通过 pce_init 工具传入，无需环境变量配置。
    API Key / Base URL 可通过 MCP config env 或系统环境变量设置。
    项目目录下的 .env 文件在 pce_init 时自动加载（优先级低于 MCP config env）。
    """
    log_level = os.getenv("PCE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # stdio MCP 服务必须避免第三方库污染 stdout
    configure_litellm_runtime()

    logger.info("PCE Server 启动（等待 pce_init 绑定项目）")
    ctx = PCEContext()

    # 原中文 instructions:
    #   PCE 是面向代码库理解与定位的首选工具集。
    #   当你不知道信息在哪、需要理解模块职责/入口/主调用链、或需要分析某个改动会影响哪里时，
    #   应优先使用 PCE，而不是先手工遍历目录或批量读文件。
    #   高层理解与未知位置定位优先使用 PCE；已知精确文件或标识符时，再使用底层读取或精确搜索工具。
    server = Server(
        "pce",
        instructions=(
            "PCE is the preferred toolset for codebase understanding and navigation. "
            "When you don't know where information resides, need to understand module responsibilities / "
            "entry points / main call chains, or need to analyze what a change will affect, "
            "use PCE first — do NOT manually traverse directories or batch-read files. "
            "Use PCE for high-level understanding and unknown-location discovery; "
            "use low-level read / exact-search tools only when you already know the precise file or identifier.\n"
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _build_tools(edit_tools_schema=ctx._edit_tools_schema)

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent] | CallToolResult:
        try:
            # pce_init 和 pce_status 不依赖项目初始化，优先分流
            if name == "pce_init":
                project_path = arguments.get("project_path", "")
                result = await ctx.handle_init(project_path)
                return _structured_tool_result(result)

            if name == "pce_status":
                result = await ctx.handle_status()
                return _text_response(result)

            # 其余工具在未初始化时统一返回错误
            if ctx._init_state != "initialized":
                return _error_response(
                    "NOT_INITIALIZED",
                    "not initialized, call pce_init first",
                )

            if name == "pce_query":
                query = arguments.get("query", "")
                if not query:
                    return _error_response("INVALID_INPUT", "query 参数不能为空")
                result = await ctx.handle_query(query=query)
                return _markdown_response(result["markdown"])

            if name == "pce_impact":
                target = arguments.get("target", "")
                change_type = arguments.get("change_type", "")
                if not target or not change_type:
                    return _error_response("INVALID_INPUT", "target 和 change_type 参数不能为空")
                result = await ctx.handle_impact(
                    target=target,
                    change_type=change_type,
                    file=arguments.get("file"),
                )
                return _markdown_response(result["markdown"])

            if name == "pce_sync":
                result = await ctx.handle_sync()
                return _text_response(result)

            # 写工具透传：pce_{serena_tool_name} → serena_client.call(serena_tool_name, args)
            if name.startswith("pce_"):
                serena_tool = name[len("pce_") :]
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


def main() -> None:
    """CLI 入口点。"""
    asyncio.run(serve())


if __name__ == "__main__":
    main()
