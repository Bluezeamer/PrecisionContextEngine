"""MockToolProvider — 用于离线测试的工具提供者。

支持两种模式：
1. 预录回放（replay）：从 JSON 文件按顺序回放录制的工具调用结果
2. 规则匹配（rules）：对 Serena 只读工具返回结构化的伪造结果

同时提供 RecordingProxy，可包装真实 ToolProvider 录制交互数据。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pce.agent_runtime.contracts import SpawnErrorCode
from pce.serena_client import SerenaToolError

logger = logging.getLogger(__name__)

# ============================================================================
# 默认 tools_schema（OpenAI function calling 格式，精简版）
# ============================================================================

# 与 SerenaClient._tool_to_openai_schema 输出格式一致
MINIMAL_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "按符号名/路径模式查找定义。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_path_pattern": {
                        "type": "string",
                        "description": "符号名称或路径模式",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "限定搜索范围的相对路径",
                    },
                    "include_body": {
                        "type": "boolean",
                        "description": "是否包含符号体源码",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "子符号检索深度",
                    },
                    "substring_matching": {
                        "type": "boolean",
                        "description": "是否启用子串匹配",
                    },
                },
                "required": ["name_path_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_referencing_symbols",
            "description": "查找引用指定符号的所有位置。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_path": {
                        "type": "string",
                        "description": "目标符号的名称路径",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "目标符号所在文件路径",
                    },
                },
                "required": ["name_path", "relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_symbols_overview",
            "description": "获取文件的符号概览（类、函数等顶层定义）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "目标文件的相对路径",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "子符号检索深度",
                    },
                },
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_for_pattern",
            "description": "在代码中搜索正则模式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "substring_pattern": {
                        "type": "string",
                        "description": "正则表达式模式",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "限定搜索范围的相对路径",
                    },
                    "context_lines_before": {
                        "type": "integer",
                        "description": "匹配前显示的上下文行数",
                    },
                    "context_lines_after": {
                        "type": "integer",
                        "description": "匹配后显示的上下文行数",
                    },
                },
                "required": ["substring_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "目标目录的相对路径",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "是否递归扫描子目录",
                    },
                    "skip_ignored_files": {
                        "type": "boolean",
                        "description": "是否跳过被忽略的文件",
                    },
                },
                "required": ["relative_path", "recursive"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "按文件名 glob 模式查找文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_mask": {
                        "type": "string",
                        "description": "文件名或通配模式",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "搜索起始目录",
                    },
                },
                "required": ["file_mask"],
            },
        },
    },
]


# ============================================================================
# 规则匹配：每个工具返回与真实 Serena 格式一致的伪造结果
# ============================================================================

RuleHandler = Callable[[dict[str, Any]], Any]


def _rule_find_symbol(args: dict[str, Any]) -> dict[str, Any]:
    """模拟 find_symbol 返回。

    返回已解析的 Python 对象，与 SerenaClient.call() 经 _jsonable 处理后的格式一致。
    """
    name = args.get("name_path_pattern", "UnknownSymbol")
    path = args.get("relative_path", "mock/module.py")
    body = ""
    if args.get("include_body"):
        body = f"def {name.split('/')[-1]}():\n    pass"
    entry: dict[str, Any] = {
        "name_path": name,
        "kind": "Function",
        "relative_path": path,
        "body_location": {"start_line": 1, "end_line": 10},
    }
    if body:
        entry["body"] = body
    return {"result": [entry]}


def _rule_find_referencing_symbols(args: dict[str, Any]) -> dict[str, Any]:
    name = args.get("name_path", "UnknownSymbol")
    path = args.get("relative_path", "mock/module.py")
    return {
        "result": [
            {
                "name_path": f"caller_of_{name}",
                "kind": "Function",
                "relative_path": path,
                "body_location": {"start_line": 20, "end_line": 25},
                "snippet": f"    result = {name}()",
            }
        ]
    }


def _rule_get_symbols_overview(args: dict[str, Any]) -> dict[str, Any]:
    path = args.get("relative_path", "mock/module.py")
    return {
        "result": {
            "Function": ["main", "helper"],
            "Class": [{"MyClass": {"Method": ["__init__", "run"]}}],
        }
    }


def _rule_search_for_pattern(args: dict[str, Any]) -> dict[str, Any]:
    pattern = args.get("substring_pattern", "")
    path = args.get("relative_path", "")
    return {
        "result": {
            f"{path or 'mock/module.py'}": [
                {"line": 10, "content": f"    # 包含 {pattern} 的匹配行"}
            ]
        }
    }


def _rule_list_dir(args: dict[str, Any]) -> dict[str, Any]:
    root = args.get("relative_path", ".")
    return {
        "result": {
            "directories": ["src", "tests", "docs"],
            "files": ["README.md", "pyproject.toml", "setup.py"],
        }
    }


def _rule_find_file(args: dict[str, Any]) -> dict[str, Any]:
    mask = args.get("file_mask", "*")
    root = args.get("relative_path", ".")
    return {
        "result": [f"{root}/mock_match_{mask}"]
    }


def _rule_spawn_agent(args: dict[str, Any]) -> dict[str, Any]:
    """模拟 spawn_agent 返回结构，供 agent_playground 等离线场景使用。

    注意：spawn_agent 在 agent.py 层直接分拣处理，不经过 MockToolProvider.call()；
    此规则仅在需要通过 MockToolProvider 测试 spawn 工具 schema 验证或 playground 时生效。
    """
    task = str(args.get("task", "")).strip()
    try:
        allocated_seconds = float(args.get("allocated_seconds", 0.0))
    except (TypeError, ValueError):
        allocated_seconds = 0.0

    if not task or allocated_seconds <= 0:
        return {
            "ok": False,
            "task_id": "mock-spawn-invalid",
            "status": "failed",
            "answer": "",
            "confidence": "low",
            "elapsed_seconds": 0.0,
            "error_code": SpawnErrorCode.SUBAGENT_INVALID_ARGS.value,
            "error_message": "spawn_agent 参数无效（mock rule）",
        }

    return {
        "ok": True,
        "task_id": "mock-spawn-ok",
        "status": "completed",
        "answer": f"mock 子任务完成: {task}",
        "confidence": "medium",
        "elapsed_seconds": 0.01,
        "error_code": None,
        "error_message": None,
    }


DEFAULT_RULES: dict[str, RuleHandler] = {
    "find_symbol": _rule_find_symbol,
    "find_referencing_symbols": _rule_find_referencing_symbols,
    "get_symbols_overview": _rule_get_symbols_overview,
    "search_for_pattern": _rule_search_for_pattern,
    "list_dir": _rule_list_dir,
    "find_file": _rule_find_file,
    "spawn_agent": _rule_spawn_agent,
}


# ============================================================================
# 录制文件格式
# ============================================================================


@dataclass
class ToolCallRecord:
    """单次工具调用的录制记录。"""

    tool_name: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_ms: int = 0


@dataclass
class _CallStats:
    """工具调用统计。"""

    total: int = 0
    errors: int = 0
    replay_hits: int = 0
    rule_hits: int = 0
    misses: int = 0
    per_tool: dict[str, int] = field(default_factory=dict)

    def record(self, tool_name: str, *, error: bool = False, source: str = "") -> None:
        self.total += 1
        if error:
            self.errors += 1
        if source == "replay":
            self.replay_hits += 1
        elif source == "rule":
            self.rule_hits += 1
        elif source == "miss":
            self.misses += 1
        self.per_tool[tool_name] = self.per_tool.get(tool_name, 0) + 1


# ============================================================================
# MockToolProvider
# ============================================================================


class MockToolProvider:
    """基于预录回放或规则匹配的离线工具提供者，满足 ToolProvider 协议。

    Args:
        project_path: 模拟的项目根路径
        tools_schema: 工具 schema 列表（默认使用 MINIMAL_TOOLS_SCHEMA）
        recording_path: 预录文件路径（JSON 格式）
        rules: 规则处理器字典（默认使用 DEFAULT_RULES）
        fallback_to_rules: 预录未命中时是否降级到规则匹配
        strict: 所有策略未命中时是否抛出异常（False 则返回空结果）
    """

    def __init__(
        self,
        project_path: str | Path,
        tools_schema: list[dict[str, Any]] | None = None,
        recording_path: str | Path | None = None,
        rules: dict[str, RuleHandler] | None = None,
        fallback_to_rules: bool = True,
        strict: bool = False,
    ) -> None:
        self._project_path = Path(project_path).resolve()
        self._tools_schema = tools_schema or list(MINIMAL_TOOLS_SCHEMA)
        self._rules = rules if rules is not None else dict(DEFAULT_RULES)
        self._fallback_to_rules = fallback_to_rules
        self._strict = strict
        self._call_log: list[ToolCallRecord] = []
        self._stats = _CallStats()

        # 加载预录数据
        self._replay_queue: list[dict[str, Any]] = []
        if recording_path is not None:
            self._load_recording(Path(recording_path))

    # ── ToolProvider 协议实现 ────────────────────────────────────────────

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        return list(self._tools_schema)

    @property
    def connected(self) -> bool:
        return True

    @property
    def project_path(self) -> Path:
        return self._project_path

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        start = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        source = ""
        try:
            result, source = self._dispatch(tool_name, args)
            return result
        except Exception as exc:
            error = exc
            if not isinstance(exc, SerenaToolError):
                raise SerenaToolError(str(exc)) from exc
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._stats.record(
                tool_name, error=error is not None, source=source or "miss"
            )
            self._call_log.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    error=str(error) if error else None,
                    duration_ms=duration_ms,
                )
            )

    # ── 观测接口 ────────────────────────────────────────────────────────

    @property
    def call_log(self) -> list[ToolCallRecord]:
        return list(self._call_log)

    @property
    def stats(self) -> dict[str, Any]:
        s = self._stats
        return {
            "total": s.total,
            "errors": s.errors,
            "replay_hits": s.replay_hits,
            "rule_hits": s.rule_hits,
            "misses": s.misses,
            "per_tool": dict(s.per_tool),
        }

    # ── 内部调度 ────────────────────────────────────────────────────────

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, str]:
        """尝试从 replay → rules → fallback 分发工具调用。返回 (result, source)。"""
        # 1. 尝试预录回放
        if self._replay_queue:
            entry = self._replay_queue[0]
            if entry.get("tool_name") == tool_name:
                self._replay_queue.pop(0)
                if entry.get("error"):
                    raise SerenaToolError(entry["error"])
                return entry.get("result"), "replay"
            # tool_name 不匹配：跳过该条目以避免队列卡死
            if self._fallback_to_rules:
                logger.warning(
                    "预录数据 tool_name 不匹配: 期望 %s, 实际 %s, 跳过该条目并降级到规则",
                    entry.get("tool_name"),
                    tool_name,
                )
                self._replay_queue.pop(0)
            else:
                raise SerenaToolError(
                    f"预录数据不匹配: 期望 {entry.get('tool_name')}, 实际 {tool_name}"
                )

        # 2. 尝试规则匹配
        if tool_name in self._rules:
            result = self._rules[tool_name](args)
            return result, "rule"

        # 3. 未命中
        if self._strict:
            raise SerenaToolError(f"未知工具且 strict 模式: {tool_name}")

        logger.warning("MockToolProvider: 未命中工具 %s，返回空结果", tool_name)
        return {"result": f"mock: 未知工具 {tool_name}"}, "miss"

    def _load_recording(self, path: Path) -> None:
        """加载预录文件。格式：{"meta": {...}, "tools_schema": [...], "calls": [...]}"""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SerenaToolError(f"录制文件不存在: {path}") from exc
        except json.JSONDecodeError as exc:
            raise SerenaToolError(f"录制文件 JSON 格式错误: {path}") from exc

        # 兼容两种格式：标准格式（带 meta + calls）和简易列表
        if isinstance(data, list):
            self._replay_queue = data
        elif isinstance(data, dict):
            if "tools_schema" in data and self._tools_schema == MINIMAL_TOOLS_SCHEMA:
                self._tools_schema = data["tools_schema"]
            self._replay_queue = data.get("calls", [])
        else:
            raise SerenaToolError(f"录制文件格式错误: {path}")

        logger.info("加载录制文件: %s (%d 条记录)", path, len(self._replay_queue))


# ============================================================================
# RecordingProxy — 包装真实 ToolProvider，录制工具调用
# ============================================================================


class RecordingProxy:
    """透明代理，拦截 call() 并记录到列表，支持导出为录制文件。

    用法：
        proxy = RecordingProxy(real_serena_client)
        # 将 proxy 传给 agent 代替 real_serena_client
        # 完成后调用 proxy.save("recording.json")
    """

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._call_log: list[ToolCallRecord] = []
        self._stats = _CallStats()

    # ── ToolProvider 协议实现（代理到真实 provider）──────────────────────

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        return self._provider.tools_schema

    @property
    def connected(self) -> bool:
        return self._provider.connected

    @property
    def project_path(self) -> Path:
        return self._provider.project_path

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        start = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = await self._provider.call(tool_name, args)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            self._stats.record(tool_name, error=error is not None)
            self._call_log.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    args=args,
                    result=result,
                    error=str(error) if error else None,
                    duration_ms=duration_ms,
                )
            )

    # ── 观测接口 ────────────────────────────────────────────────────────

    @property
    def call_log(self) -> list[ToolCallRecord]:
        return list(self._call_log)

    @property
    def stats(self) -> dict[str, Any]:
        s = self._stats
        return {
            "total": s.total,
            "errors": s.errors,
            "per_tool": dict(s.per_tool),
        }

    # ── 录制导出 ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """将录制数据保存为 JSON 文件。"""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        recording = {
            "meta": {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "project_path": str(self._provider.project_path),
            },
            "tools_schema": self._provider.tools_schema,
            "calls": [
                {
                    "tool_name": r.tool_name,
                    "args": r.args,
                    "result": r.result,
                    "error": r.error,
                }
                for r in self._call_log
            ],
        }
        out_path.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("录制已保存: %s (%d 条)", out_path, len(self._call_log))
