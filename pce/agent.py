"""PCE Agent — 手工实现的 ReAct (Reasoning + Acting) 循环。

基于 litellm 的 tool_call 流程驱动 Serena 工具调用,实现:
- 自然语言代码库查询 (pce_query)
- 变更影响边界分析 (pce_impact)

会话管理:
- in-memory 存储 `_sessions: dict[str, list[dict]]`
- 循环以挂钟时间（而非步数）为终止条件
- 上下文接近窗口上限时触发 compact：将已有认知蒸馏为摘要，重建对话窗口后继续
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

# 认知确认回调：接受文件路径列表，执行暂存区标记
AcknowledgeCallback = Callable[[list[str]], Awaitable[None]]

import aiofiles
import litellm
import litellm.exceptions as litellm_exc

from .agent_runtime.contracts import (
    MAX_SPAWNS_PER_LOOP,
    SPAWN_AGENT_TOOL,
    SpawnErrorCode,
    SpawnRequest,
    SpawnResult,
    SpawnStatus,
)
from .agent_runtime.spawner import invoke_spawn
from .insight_cache import InsightCache
from .models import ImpactResponse, InsightConfidence, QueryResponse, ReferenceEdge, SymbolRef
from .serena_client import SerenaClient, SerenaClientError
from ._env import build_litellm_model, get_completion_overrides, get_env_text

logger = logging.getLogger(__name__)

MAX_SECONDS = 600  # 默认推理时间上限（秒），10 分钟
# 上下文窗口触发 compact 的使用率阈值（预留 20% 用于 compact 操作本身）
_COMPACT_THRESHOLD = 0.80
# step-3.5-flash 的上下文窗口大小（token 数），可通过环境变量覆盖
_CONTEXT_WINDOW = int(os.getenv("PCE_CONTEXT_WINDOW", "256000"))
# 通过 PCE_PROVIDER + PCE_MODEL 环境变量配置（必填），运行时读取。
# 示例:
#   PCE_PROVIDER=openrouter  PCE_MODEL=openai/gpt-4o-mini
#   PCE_PROVIDER=openai      PCE_MODEL=gpt-4o-mini
#   PCE_PROVIDER=anthropic   PCE_MODEL=claude-3-haiku-20240307


# ---------------------------------------------------------------------------
# 模型降级路由（fallback chain）
# ---------------------------------------------------------------------------


def _parse_model_fallbacks(raw: str) -> list[str]:
    """解析逗号分隔的 fallback 模型列表，按出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in seen:
            seen.add(model)
            result.append(model)
    return result


class LLMCompletionError(RuntimeError):
    """所有候选模型均失败时抛出，携带完整降级记录。"""

    def __init__(self, attempts: list[dict[str, str]]) -> None:
        self.attempts = attempts
        self.models = [a["model"] for a in attempts]
        parts = [f'{a["model"]}({a["error_type"]}): {a["reason"]}' for a in attempts]
        super().__init__(
            f"LLM fallback chain exhausted; models={self.models}; " + " | ".join(parts)
        )


def _should_fallback_model(exc: Exception) -> bool:
    """判断是否应切换到下一个模型（不可恢复的模型级错误）。

    仅对以下情况触发 fallback，不在同一模型上重复重试（litellm 已重试过）：
    - 限流（429）
    - 鉴权/权限（401/403）
    - 模型不存在（404 或特征文本）
    """
    # 优先使用 isinstance 判断（稳定、不受类名重命名影响）
    if isinstance(
        exc,
        (
            litellm_exc.RateLimitError,
            litellm_exc.AuthenticationError,
            litellm_exc.PermissionDeniedError,
            litellm_exc.NotFoundError,
        ),
    ):
        return True
    if isinstance(exc, (litellm_exc.BadRequestError, litellm_exc.InvalidRequestError)):
        msg = _stringify_error(exc).lower()
        return any(
            k in msg
            for k in (
                "model not found",
                "unknown model",
                "invalid model",
                "does not exist",
                "no deployments available",
            )
        )
    return False


def _stringify_error(exc: Exception) -> str:
    """提取稳定、可读的异常原因字符串。"""
    return str(getattr(exc, "message", None) or exc or type(exc).__name__)


def _extract_finish_reason(response: Any) -> str:
    """从 litellm 响应中提取 finish_reason，缺失时默认 'stop'。"""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])
    if choices:
        choice = choices[0]
        raw = (
            getattr(choice, "finish_reason", None)
            if not isinstance(choice, dict)
            else choice.get("finish_reason")
        )
        if raw:
            return str(raw)
    return "stop"


SYSTEM_PROMPT_HEADER = """\
你是 PCE (Precision Context Engine) 的核心 Agent。

职责:
1. 使用 Serena MCP 工具检索代码结构、符号定义与引用关系
2. 在有限上下文中给出可靠的代码理解与影响分析结论
3. 严格基于检索证据回答,不编造任何不存在的信息

能力限制:
- 只能使用提供的 Serena 工具和 Memory 中的索引内容
- 不生成实际代码,不执行写操作
- 如果信息不足,明确说明并建议下一步检索方向

**强制流程**:
- 回答任何关于项目代码的问题前,必须先通过 tool_calls 调用 Serena 工具获取证据
- 若已获取的证据不足,继续调用工具,不要凭推测作答
- 完成推理后,必须调用 deliver 工具提交最终结论,禁止直接以文字形式回答

**spawn_agent 使用策略**:
spawn_agent 将一个"可独立求解"的子问题委托给子 Agent 推理，子 Agent 拥有独立上下文，
完成后将核心结论以 tool result 形式返回。它不是默认路径，而是隔离复杂子任务的手段。

适合 spawn 的判断标准（核心）：子问题可在不依赖主线历史的情况下独立求解，
且自身需要多步工具检索，放在主线会显著分散推理焦点。

推荐场景:
- 需要多步追踪的局部分析（如：先定位符号再追所有引用）
- 平行的子模块调查（如：同时了解 A 模块和 B 模块的实现）
- 预计证据收集耗时较长、结果可被主线直接消费的任务

不要使用 spawn_agent 的场景:
- 一两次 Serena 工具调用就能完成的简单查询——直接调工具更快
- 需要依赖主线全局上下文才能作答的任务
- 时间预算所剩不多时，应立即整合已有信息直接 deliver
- spawn 次数已接近上限（上限为 3 次），优先留给最复杂的子任务

处理 spawn_agent 返回结果:
- ok=true：将 answer 作为待整合的子结论，必要时补充工具验证后并入主线
- ok=false：不要中断；根据 error_code 降级为直接调用 Serena 工具继续推理

**交付格式指引**:
上层 Agent 可在 query 中明确指定期望格式,PCE Agent 必须严格遵从。未明确指定时,按以下默认规则选择格式:

- 定位类任务(查找函数/类/变量在哪里):
  返回结构化列表,每项包含:
    file: 相对路径(如 pce/agent.py)
    line_range: [start, end]
    name_path: 符号路径(如 PCEAgent/_run_react_loop,遵循 Parent/child 格式)

- 影响分析类任务(pce_impact):
  返回含行号的引用点列表,每项包含:
    file: 相对路径
    line: 引用所在行
    referencing_symbol: 引用该符号的所属函数/方法名
    snippet: 该行附近的代码片段（上下各 2 行）
  并在列表末尾给出建议修改顺序(叶节点优先)

- 问答/理解类任务:
  自然语言回答;凡提及代码位置,必须附上 file:line 格式

输出要求:
- 回答简洁精准,优先给出文件路径和行号定位
- 影响分析时列出所有直接引用点,不要遗漏\
"""

# deliver 是一个虚拟工具,不转发给 Serena,由循环内部拦截处理
# agent 调用它表示主动声明"任务完成",是唯一的正常终止信号
DELIVER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "deliver",
        "description": "提交最终结论并结束当前任务。完成推理后必须调用此工具,不得直接以文字回答。",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "最终结论内容"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "结论置信度",
                },
            },
            "required": ["answer"],
        },
    },
}

# acknowledge_changes 是另一个虚拟工具,用于 PCE Agent 标记已探索并形成正确认知的文件
# Agent 在通过 Serena 工具读取并理解变更文件后调用此工具
ACKNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "acknowledge_changes",
        "description": (
            "标记指定文件的变更已被探索并形成正确认知。"
            "在通过 Serena 工具读取并理解变更文件的最新内容后，"
            "调用此工具更新暂存区状态。可多次调用，渐进式确认。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "已形成认知的文件相对路径列表",
                },
            },
            "required": ["paths"],
        },
    },
}

# 无 tool_calls 时的最大纠正次数(超出后强制终止,视为异常)
_MAX_NO_TOOL_RETRIES = 3
# finish_reason=length 时最大续写次数
_MAX_LENGTH_CONT = 2
# 单次 LLM 调用超时的最大重试次数
_MAX_TIMEOUT_RETRIES = 1
# Insight Cache 注入与蒸馏参数
_INSIGHT_TOP_K = 5
_INSIGHT_TOKEN_BUDGET = 4000  # 字符数上限（粗略估算 token）
_INSIGHT_MAX_SCOPES = 3  # 每次 deliver 最多写入的 scope 数量
_INSIGHT_MAX_CONTENT_CHARS = 1200  # 单条 insight content 最大字符数
# 从文本中提取形如 pce/agent.py 或 ./src/foo.py:123 的路径
_PATH_RE = re.compile(r"(?<![\w./-])(?:\./)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_+\-]+)(?::\d+)?")


# ============================================================================
# 辅助函数
# ============================================================================


def _safe_json_dumps(value: Any) -> str:
    """安全地将值序列化为 JSON 字符串。"""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _try_parse_json(value: Any) -> Any:
    """尝试将字符串解析为 JSON,失败时原样返回。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
    return value


def _extract_message(response: Any) -> dict[str, Any]:
    """从 litellm 响应中提取 choice message。"""
    # 处理 Pydantic/dataclass 格式
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices", [])

    if not choices:
        return {"role": "assistant", "content": ""}

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message", {})

    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=False)
        # 防御: model_dump 可能丢失 tool_calls 字段(某些模型返回格式差异)
        if "tool_calls" not in dumped and hasattr(message, "tool_calls"):
            tc = getattr(message, "tool_calls")
            if tc is not None:
                dumped["tool_calls"] = tc
        # 兼容旧版 function_call 格式
        if "function_call" not in dumped and hasattr(message, "function_call"):
            fc = getattr(message, "function_call")
            if fc is not None:
                dumped["function_call"] = fc
        # StepFun 等 provider 要求 assistant 消息的 content 字段必须存在
        if dumped.get("content") is None:
            dumped["content"] = ""
        return dumped
    if isinstance(message, dict):
        # 同上，确保 content 字段存在
        if message.get("content") is None:
            message = {**message, "content": ""}
        return message
    return {"role": "assistant", "content": str(message)}


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """从 message 中提取 tool_calls 列表。"""
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    # 兼容旧版 function_call 格式(某些模型/路由可能使用)
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return [{"id": str(uuid.uuid4()), "function": function_call}]
    # 兼容 function_call 为 Pydantic/dataclass 对象的情况
    if function_call is not None and hasattr(function_call, "name"):
        name = getattr(function_call, "name", None)
        if name:
            return [
                {
                    "id": str(uuid.uuid4()),
                    "function": {
                        "name": name,
                        "arguments": getattr(function_call, "arguments", None),
                    },
                }
            ]
    return []


def _parse_tool_call_args(raw_args: Any) -> dict[str, Any] | None:
    """解析 tool_call 的 arguments 字段。失败时返回 None（而非空 dict）。

    返回 None 意味着调用方应将其视为解析错误，而非"无参数调用"。
    """
    if raw_args is None:
        return None
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            result = json.loads(raw_args)
            return result if isinstance(result, dict) else None
        except (ValueError, json.JSONDecodeError):
            return None
    return None


# ============================================================================
# 响应解析
# ============================================================================


def _parse_query_response(content: str, session_id: str) -> QueryResponse:
    """将 Agent 输出解析为 QueryResponse。"""
    _FALLBACK_ANSWERS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成推理，请缩小问题范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 推理超出总时长预算，请缩小问题范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，推理异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，推理可能不完整，请重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小问题范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
        "__REACT_LLM_EXHAUSTED__": "Agent 的模型降级链已全部失败，请检查模型配置或限流状态后重试。",
    }
    if content in _FALLBACK_ANSWERS:
        return QueryResponse(
            answer=_FALLBACK_ANSWERS[content],
            evidence=[],
            related_symbols=[],
            related_files=[],
            session_id=session_id,
        )

    payload = _try_parse_json(content)

    if isinstance(payload, dict):
        answer = str(payload.get("answer") or payload.get("content") or content)

        related_symbols = []
        for item in payload.get("related_symbols") or []:
            if isinstance(item, dict):
                try:
                    related_symbols.append(SymbolRef.model_validate(item))
                except Exception:
                    continue

        related_files = [
            Path(p) for p in (payload.get("related_files") or []) if isinstance(p, str)
        ]
        evidence = [str(e) for e in (payload.get("evidence") or []) if e]

        return QueryResponse(
            answer=answer,
            evidence=evidence,
            related_symbols=related_symbols,
            related_files=related_files,
            session_id=session_id,
        )

    return QueryResponse(
        answer=str(content) if content else "未能生成有效回答",
        evidence=[],
        related_symbols=[],
        related_files=[],
        session_id=session_id,
    )


def _parse_impact_response(content: str, session_id: str) -> ImpactResponse:
    """将 Agent 输出解析为 ImpactResponse。"""
    _FALLBACK_RISKS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成影响分析，请缩小分析范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 分析超出总时长预算，请缩小分析范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，分析异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，分析可能不完整，请重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小分析范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
        "__REACT_LLM_EXHAUSTED__": "Agent 的模型降级链已全部失败，请检查模型配置或限流状态后重试。",
    }
    if content in _FALLBACK_RISKS:
        return ImpactResponse(
            impact_chain=[],
            boundary=[],
            risks=[_FALLBACK_RISKS[content]],
            unknowns=[],
            session_id=session_id,
        )

    payload = _try_parse_json(content)

    if isinstance(payload, dict):
        impact_chain = []
        for item in payload.get("impact_chain") or []:
            if isinstance(item, dict):
                try:
                    impact_chain.append(ReferenceEdge.model_validate(item))
                except Exception:
                    continue

        boundary = []
        for item in payload.get("boundary") or []:
            if isinstance(item, dict):
                try:
                    boundary.append(SymbolRef.model_validate(item))
                except Exception:
                    continue

        risks = [str(r) for r in (payload.get("risks") or []) if r]
        unknowns = [str(u) for u in (payload.get("unknowns") or []) if u]

        return ImpactResponse(
            impact_chain=impact_chain,
            boundary=boundary,
            risks=risks,
            unknowns=unknowns,
            session_id=session_id,
        )

    # 无法解析为结构化格式,将原始内容作为风险描述
    return ImpactResponse(
        impact_chain=[],
        boundary=[],
        risks=[str(content)] if content else ["分析结果无法解析"],
        unknowns=[],
        session_id=session_id,
    )


# ============================================================================
# 主 Agent 类
# ============================================================================


class PCEAgent:
    """PCE ReAct Agent。

    维护会话状态,驱动 Serena 工具调用,返回推理结论。
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        model_fallbacks: list[str] | None = None,
        max_seconds: float = MAX_SECONDS,
        insight_cache: InsightCache | None = None,
    ) -> None:
        explicit_model = model.strip() if model else None
        explicit_provider = provider.strip() if provider else None

        if explicit_model is None:
            self._provider = get_env_text("PCE_PROVIDER")
            self._model = get_env_text("PCE_MODEL")
            if not self._provider or not self._model:
                raise ValueError(
                    "未配置 PCE_PROVIDER / PCE_MODEL，请通过 MCP config env、系统环境变量"
                    "或项目 .env 设置，例如 PCE_PROVIDER=openrouter, "
                    "PCE_MODEL=openai/gpt-4o-mini。"
                )
        else:
            # 显式传参保留灵活性：用于测试或调试脚本时，可直接传完整 LiteLLM model。
            self._provider = explicit_provider
            self._model = explicit_model

        # 降级链：去掉与主模型相同的候选项
        raw_fallbacks = (
            model_fallbacks
            if model_fallbacks is not None
            else _parse_model_fallbacks(os.getenv("PCE_MODEL_FALLBACKS", ""))
        )
        self._model_fallbacks = [m for m in raw_fallbacks if m and m != self._model]
        self._max_seconds = max_seconds
        self._deliver_tool = DELIVER_TOOL
        self._insight_cache = insight_cache
        # in-memory 会话存储,进程重启后丢失(MVP 范围内可接受)
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    # =========================================================================
    # Insight 蒸馏辅助
    # =========================================================================

    @staticmethod
    def _confidence_from_str(value: str | None) -> InsightConfidence:
        """将 deliver 返回的置信度字符串映射为 InsightConfidence 枚举。"""
        if value == "high":
            return InsightConfidence.HIGH
        if value == "low":
            return InsightConfidence.LOW
        return InsightConfidence.MEDIUM

    def _extract_path_candidates(self, text: str) -> list[str]:
        """从文本中提取候选相对路径（去重，保留出现顺序）。"""
        seen: set[str] = set()
        result: list[str] = []
        for m in _PATH_RE.finditer(text):
            candidate = m.group(1).removeprefix("./")
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
        return result

    def _pick_insight_scopes(self, answer: str, question: str) -> list[str]:
        """从 answer + question 中选出真实存在于项目根目录下的相对文件路径。

        优先取 answer 中出现的路径（更贴近结论），再回退到 question。
        最多返回 _INSIGHT_MAX_SCOPES 个。
        """
        if self._insight_cache is None:
            return []
        root = self._insight_cache.project_root
        ordered = self._extract_path_candidates(answer) + self._extract_path_candidates(question)

        scopes: list[str] = []
        seen: set[str] = set()
        for candidate in ordered:
            try:
                resolved = (root / candidate).resolve()
                rel = resolved.relative_to(root).as_posix()
            except (ValueError, OSError):
                continue
            if not resolved.is_file():
                continue
            if rel in seen:
                continue
            seen.add(rel)
            scopes.append(rel)
            if len(scopes) >= _INSIGHT_MAX_SCOPES:
                break
        return scopes

    @staticmethod
    def _distill_insight_content(question: str, answer: str, scope: str) -> str:
        """将 question/answer 对蒸馏为与 scope 相关的紧凑 insight 内容。"""
        # 问题截断
        q = " ".join(question.strip().split())
        if len(q) > 220:
            q = q[:220].rstrip() + "..."

        # 优先提取 answer 中包含 scope 的行作为核心结论
        lines = [ln.strip() for ln in answer.splitlines() if scope in ln and ln.strip()][:8]
        if lines:
            core = "\n".join(f"- {ln}" for ln in lines)
        else:
            # 回退：取 answer 全文（空白折叠）
            core = " ".join(answer.strip().split())

        content = f"问题: {q}\n与 {scope} 相关结论:\n{core}"
        if len(content) > _INSIGHT_MAX_CONTENT_CHARS:
            content = content[:_INSIGHT_MAX_CONTENT_CHARS].rstrip() + "..."
        return content

    async def _persist_insights(
        self,
        question: str,
        answer: str,
        confidence: str | None,
    ) -> None:
        """将 deliver 结果蒸馏后写入 InsightCache。

        仅在配置了 insight_cache 且 answer 不是异常兜底字符串时执行。
        失败不抛出，仅记录警告。
        """
        if self._insight_cache is None:
            return
        if answer.startswith("__REACT_"):
            return

        scopes = self._pick_insight_scopes(answer, question)
        if not scopes:
            return

        conf = self._confidence_from_str(confidence)
        for scope in scopes:
            content = self._distill_insight_content(question, answer, scope)
            try:
                await self._insight_cache.upsert(scope=scope, content=content, confidence=conf)
                logger.debug(f"Insight 蒸馏写入: scope={scope}")
            except FileNotFoundError:
                logger.debug(f"Insight scope 文件不存在，跳过: {scope}")
            except Exception as e:
                logger.warning(f"Insight upsert 失败: scope={scope}: {e}")

    async def _read_pce_file(self, path: Path) -> str:
        """读取 .pce 目录下的文件内容,文件不存在时返回占位文本。"""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return f"(未找到 {path.name},索引尚未构建)"
        except Exception as e:
            logger.warning(f"读取文件失败: {path}: {e}")
            return f"(读取 {path.name} 失败)"

    async def _build_system_prompt(self, memory_root: Path | None) -> str:
        """构建 system prompt,注入 structure.md 和 annotations.md 内容。

        若 insight_cache 已配置，在末尾追加动态认知块（top-k 条目）。
        """
        root = memory_root or Path.cwd()
        pce_dir = root / ".pce"

        structure_md = await self._read_pce_file(pce_dir / "structure.md")
        annotations_md = await self._read_pce_file(pce_dir / "annotations.md")

        sections = [
            SYSTEM_PROMPT_HEADER,
            "",
            "## 项目结构 (structure.md)",
            structure_md.strip(),
            "",
            "## 语义注解 (annotations.md)",
            annotations_md.strip(),
        ]

        if self._insight_cache is not None:
            try:
                injected, _ = await self._insight_cache.get_top_k(
                    k=_INSIGHT_TOP_K,
                    token_budget=_INSIGHT_TOKEN_BUDGET,
                )
                if injected.strip():
                    sections += [
                        "",
                        "## 动态认知提示",
                        "以下内容来自历史会话蒸馏，可能过时；必须通过工具验证后再使用。",
                        "",
                        injected.strip(),
                    ]
            except Exception as e:
                logger.warning(f"读取 Insight Cache 失败，忽略本轮注入: {e}")

        return "\n".join(sections)

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[Any, str]:
        """调用 litellm.completion，返回 (response, finish_reason)。

        支持模型降级路由：当主模型遇到不可恢复的模型级错误（限流、鉴权、模型不存在等）时，
        依次尝试 fallback 模型。litellm 本身已有重试机制，此处不对同一模型重复重试。

        finish_reason 取值: "stop" | "tool_calls" | "length" | 其他
        缺失时默认 "stop"。

        Raises:
            asyncio.TimeoutError: 单步超时（由上层 _run_react_loop 处理）
            LLMCompletionError: 所有候选模型均失败（fallback chain 非空时）
            其他 litellm 异常: fallback chain 为空时原样透传
        """
        # litellm.Timeout 不是 asyncio.TimeoutError 的子类，统一转换
        _timeout_types = (asyncio.TimeoutError, litellm_exc.Timeout)
        completion_overrides = get_completion_overrides()

        model_chain: list[str] = []
        for model in [self._model, *self._model_fallbacks]:
            model_chain.append(build_litellm_model(self._provider, model))

        # fallback 为空时保持原行为：直接调用，异常原样透传
        if len(model_chain) == 1:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        litellm.completion,
                        model=model_chain[0],
                        messages=messages,
                        tools=tools if tools else None,
                        **completion_overrides,
                        temperature=0.2,
                    ),
                    timeout=60.0,
                )
            except litellm_exc.Timeout:
                raise asyncio.TimeoutError("litellm.Timeout -> asyncio.TimeoutError")
            return response, _extract_finish_reason(response)

        # 有 fallback 时：逐个尝试，记录每次失败
        attempts: list[dict[str, str]] = []
        for idx, model in enumerate(model_chain):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        litellm.completion,
                        model=model,
                        messages=messages,
                        tools=tools if tools else None,
                        **completion_overrides,
                        temperature=0.2,
                    ),
                    timeout=60.0,
                )
                if idx > 0:
                    logger.info("模型降级成功: %s (第 %d 候选)", model, idx + 1)
                return response, _extract_finish_reason(response)
            except _timeout_types:
                # 超时由上层统一处理，不纳入降级逻辑
                raise asyncio.TimeoutError("timeout in _completion")
            except Exception as exc:
                if not _should_fallback_model(exc):
                    # 非模型级错误（如网络中断、JSON 解析失败等），直接透传
                    raise

                attempts.append(
                    {
                        "model": model,
                        "error_type": type(exc).__name__,
                        "reason": _stringify_error(exc),
                    }
                )

                if idx < len(model_chain) - 1:
                    next_model = model_chain[idx + 1]
                    logger.warning(
                        "模型调用失败，触发降级: %s -> %s (%s)",
                        model,
                        next_model,
                        attempts[-1]["error_type"],
                    )
                    continue

                # 所有模型均已尝试
                logger.warning("模型降级链耗尽: %s", [a["model"] for a in attempts])
                raise LLMCompletionError(attempts) from exc

        # 理论上不可达，但类型检查需要
        raise LLMCompletionError(attempts)

    @staticmethod
    def _extract_next_prompt_size(response: Any) -> int:
        """估算下一轮调用的 prompt 大小，失败时返回 0。

        下一轮 prompt = 本轮 prompt + 本轮 completion（本轮输出会追加到历史）
        因此用 total_tokens（= prompt + completion）作为下一轮窗口占用的预测值，
        无需累加历史（累加会因每轮 prompt 已包含完整历史而产生 O(n²) 重复计数）。
        """
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            return 0

        def _get(attr: str) -> int:
            v = getattr(usage, attr, None) if not isinstance(usage, dict) else usage.get(attr)
            return int(v) if v else 0

        total = _get("total_tokens")
        if total:
            return total
        # 部分 provider 不返回 total_tokens，退而求其次手动相加
        return _get("prompt_tokens") + _get("completion_tokens")

    async def _maybe_compact(
        self,
        messages: list[dict[str, Any]],
        token_used: int,
        compact_failed: bool = False,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """在上下文使用率达到阈值时触发 compact，蒸馏认知后重建对话窗口。

        Args:
            messages: 当前对话消息列表
            token_used: 上一轮的 total_tokens（= prompt + completion，即下一轮窗口占用预测值）
            compact_failed: 本轮是否已经尝试过 compact 但失败，避免重复尝试

        Returns:
            (新 messages 列表, 更新后的 token_used, compact_failed 标记)
            若未触发 compact，原样返回传入值。
        """
        if (
            _CONTEXT_WINDOW <= 0
            or compact_failed
            or token_used / _CONTEXT_WINDOW < _COMPACT_THRESHOLD
        ):
            return messages, token_used, compact_failed

        logger.info(
            f"上下文使用率 {token_used}/{_CONTEXT_WINDOW} "
            f"({token_used / _CONTEXT_WINDOW:.1%}) 达到阈值，触发 compact"
        )

        # 提取原始用户问题（第一条 user 消息），让摘要不丢失任务目标
        system_msgs = [m for m in messages if m.get("role") == "system"]
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        task_hint = f"\n\n当前任务：{first_user['content']}" if first_user else ""

        # 仅传 system + 摘要请求，避免在 token 接近满时再塞满历史
        compact_request: list[dict[str, Any]] = system_msgs + [
            {
                "role": "user",
                "content": (
                    f"请用简洁的自然语言总结目前的推理进展，包括：\n"
                    "1. 已确认的关键事实与结论\n"
                    "2. 尚未完成的子任务及下一步行动\n"
                    "输出应足够精炼，以便在新对话窗口中直接继续推理。"
                    f"{task_hint}"
                ),
            }
        ]

        try:
            # 不传工具列表，让模型自由输出摘要文本
            summary_response, _ = await self._completion(compact_request, [])
            summary_msg = _extract_message(summary_response)
            summary_text = str(summary_msg.get("content") or "").strip()
            compact_tokens = self._extract_next_prompt_size(summary_response)
        except LLMCompletionError:
            # 模型降级链耗尽，不应被 compact 吞掉，让上层处理
            raise
        except Exception as e:
            logger.warning(f"compact 摘要生成失败，跳过压缩（本轮不再重试）: {e}")
            return messages, token_used, True  # 标记失败，本轮不再重试

        if not summary_text:
            logger.warning("compact 摘要为空，跳过压缩（本轮不再重试）")
            return messages, token_used, True

        # 重建消息列表：system prompt + 摘要注入（作为新窗口的起点）
        new_messages: list[dict[str, Any]] = system_msgs + [
            {
                "role": "user",
                "content": f"[上下文摘要 — 基于前序推理]\n{summary_text}",
            },
            {
                "role": "assistant",
                "content": "已了解前序推理摘要，继续执行剩余任务。",
            },
        ]

        logger.info(f"compact 完成，新窗口消息数: {len(new_messages)}")
        # compact 调用本身消耗了 compact_tokens，以此作为新窗口的基线；重置失败标记
        return new_messages, compact_tokens, False

    async def _run_react_loop(
        self,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
        acknowledge_cb: AcknowledgeCallback | None = None,
        depth: int = 0,
        deadline: float | None = None,
    ) -> tuple[str, str | None]:
        """执行 ReAct 循环直到 agent 调用 deliver 或超出时长预算。

        Returns:
            (answer, confidence)，正常终止时 confidence 为 deliver 提供的置信度字符串，
            异常终止时 confidence 为 None。

        终止路径:
          正常: agent 调用 deliver(answer=...)
          兜底: __REACT_TIMEOUT_BUDGET__      — 总时长预算耗尽
                __REACT_NO_TOOL_EXHAUSTED__   — 无 tool_calls 纠正次数耗尽
                __REACT_LENGTH_EXHAUSTED__    — length 截断续写次数耗尽
                __REACT_TIMEOUT__             — 单步超时重试次数耗尽
                __REACT_LLM_EXHAUSTED__       — 模型降级链耗尽
                __REACT_DELIVER_EMPTY__       — deliver 参数为空或解析失败

        计数器独立，互不干扰:
          no_tool_retries      — LLM 无 tool_calls（无论 finish_reason）
          length_continuations — finish_reason=length 触发续写
          timeout_retries      — 每步内独立，asyncio.TimeoutError
        """
        # 构建工具列表：Serena 只读工具 + 虚拟终止工具 + 可选工具
        tools_schema = serena_client.tools_schema + [self._deliver_tool]
        if depth == 0:
            # 子 Agent（depth>=1）不能递归 spawn，仅主 Agent 注入该工具
            tools_schema = tools_schema + [SPAWN_AGENT_TOOL]
        if acknowledge_cb is not None:
            tools_schema = tools_schema + [ACKNOWLEDGE_TOOL]

        no_tool_retries = 0
        length_continuations = 0
        spawn_count = 0  # 本循环内 spawn 次数，不超过 MAX_SPAWNS_PER_SESSION
        token_used = 0
        compact_failed = False
        start = time.monotonic()
        # 使用绝对 deadline，支持父子 Agent 共享同一时间轴
        if deadline is None:
            deadline = start + self._max_seconds

        while True:
            # ── 时间预算检查 ──────────────────────────────────────────────────
            if time.monotonic() >= deadline:
                logger.warning(f"ReAct 循环超出截止时间（depth={depth}），强制终止")
                return "__REACT_TIMEOUT_BUDGET__", None

            # ── 上下文 compact 检查 ───────────────────────────────────────────
            try:
                messages, token_used, compact_failed = await self._maybe_compact(
                    messages, token_used, compact_failed
                )
            except LLMCompletionError as e:
                logger.warning("compact 阶段模型降级链耗尽，终止 ReAct 循环: %s", e)
                return "__REACT_LLM_EXHAUSTED__", None

            # ── LLM 调用，内嵌超时重试（每步独立计数）────────────────────────
            timeout_retries = 0
            while True:
                try:
                    response, finish_reason = await self._completion(messages, tools_schema)
                    break
                except LLMCompletionError as e:
                    logger.warning("模型降级链耗尽，终止 ReAct 循环: %s", e)
                    return "__REACT_LLM_EXHAUSTED__", None
                except asyncio.TimeoutError:
                    timeout_retries += 1
                    if timeout_retries <= _MAX_TIMEOUT_RETRIES:
                        logger.warning(f"模型调用超时(第 {timeout_retries} 次),正在重试当前步骤")
                        continue
                    logger.warning("模型调用超时，重试次数耗尽，强制终止")
                    return "__REACT_TIMEOUT__", None

            token_used = self._extract_next_prompt_size(response)

            message = _extract_message(response)
            if "role" not in message:
                message["role"] = "assistant"

            tool_calls = _extract_tool_calls(message)

            # ── finish_reason=length：输出被截断 ─────────────────────────────
            # 256K 上下文下概率极低，但仍需兜底处理
            if finish_reason == "length":
                length_continuations += 1
                if length_continuations > _MAX_LENGTH_CONT:
                    logger.warning("输出截断续写次数耗尽，强制终止")
                    return "__REACT_LENGTH_EXHAUSTED__", None
                if message.get("content"):
                    # 推理文字被截断：保留已有内容，追加"请继续"
                    messages.append(message)
                    messages.append({"role": "user", "content": "请继续"})
                    logger.warning(f"输出被截断(content,第 {length_continuations} 次),追加续写指令")
                else:
                    # tool_call JSON 被截断：追加空占位 assistant 保持对话结构连续，
                    # 再追加 user 纠正，避免产生"孤立 user 消息"的非法序列
                    messages.append({"role": "assistant", "content": ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": "刚才的工具调用格式不完整，请重新完整地调用工具。",
                        }
                    )
                    logger.warning(
                        f"输出被截断(tool_call,第 {length_continuations} 次),要求重新调用"
                    )
                continue

            # 正常路径：将 assistant 消息追加到历史
            messages.append(message)

            # ── 无 tool_calls：LLM 违反约束（无论 finish_reason）────────────
            # 统一处理，避免 finish_reason=tool_calls 但实际 tool_calls 为空的漏网情况
            if not tool_calls:
                no_tool_retries += 1
                if no_tool_retries <= _MAX_NO_TOOL_RETRIES:
                    logger.warning(
                        f"无 tool_calls 输出(第 {no_tool_retries} 次),"
                        f"finish_reason={finish_reason},追加强化纠正指令"
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "你刚才直接用文字回答，这违反了强制流程。"
                                "必须先通过 tool_calls 调用工具获取证据，"
                                "完成推理后再用 deliver 提交最终结论，禁止直接回答。"
                            ),
                        }
                    )
                else:
                    logger.warning("无 tool_calls 纠正次数耗尽，强制终止循环")
                    return "__REACT_NO_TOOL_EXHAUSTED__", None
                continue

            # ── 分拣 deliver、acknowledge、spawn 与 Serena 工具调用 ────────────
            deliver_call = None
            acknowledge_calls = []
            spawn_calls = []
            serena_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name") or tc.get("name")
                else:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", None) or getattr(tc, "name", None)
                if name == "deliver":
                    deliver_call = tc
                elif name == "acknowledge_changes":
                    acknowledge_calls.append(tc)
                elif name == "spawn_agent":
                    spawn_calls.append(tc)
                else:
                    serena_calls.append(tc)

            # 先并行执行所有 Serena 工具（确保工具结果不丢失）
            if serena_calls:
                tool_results = await asyncio.gather(
                    *[self._invoke_tool(tc, serena_client) for tc in serena_calls]
                )
                for result_msg in tool_results:
                    messages.append({"role": "tool", **result_msg})

            # ── 处理 spawn_agent 调用（串行，避免子任务并发挤占预算）────────────
            for tc in spawn_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    tc_id = tc.get("id") or str(uuid.uuid4())
                    raw_args = fn.get("arguments") or tc.get("arguments")
                else:
                    fn = getattr(tc, "function", None)
                    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())
                    raw_args = getattr(fn, "arguments", None)

                args = _parse_tool_call_args(raw_args)
                if args is None:
                    spawn_result = SpawnResult(
                        ok=False,
                        task_id=str(uuid.uuid4()),
                        status=SpawnStatus.FAILED,
                        answer="",
                        confidence="low",
                        elapsed_seconds=0.0,
                        error_code=SpawnErrorCode.SUBAGENT_INVALID_ARGS,
                        error_message="spawn_agent 参数解析失败：无效 JSON",
                    )
                elif spawn_count >= MAX_SPAWNS_PER_LOOP:
                    spawn_result = SpawnResult(
                        ok=False,
                        task_id=str(uuid.uuid4()),
                        status=SpawnStatus.FAILED,
                        answer="",
                        confidence="low",
                        elapsed_seconds=0.0,
                        error_code=SpawnErrorCode.SUBAGENT_INVALID_ARGS,
                        error_message=(
                            f"本次循环 spawn 次数已达上限（max={MAX_SPAWNS_PER_LOOP}），"
                            "请直接调用 Serena 工具或整合已有信息后 deliver。"
                        ),
                    )
                else:
                    try:
                        request = SpawnRequest(
                            task=str(args.get("task", "")).strip(),
                            allocated_seconds=float(args.get("allocated_seconds", 0.0)),
                            expected_output=str(args.get("expected_output", "") or ""),
                            context=(
                                args["context"] if isinstance(args.get("context"), dict) else {}
                            ),
                            strict=bool(args.get("strict", False)),
                        )
                        if not request.task:
                            raise ValueError("task 不能为空")
                    except Exception as exc:
                        spawn_result = SpawnResult(
                            ok=False,
                            task_id=str(uuid.uuid4()),
                            status=SpawnStatus.FAILED,
                            answer="",
                            confidence="low",
                            elapsed_seconds=0.0,
                            error_code=SpawnErrorCode.SUBAGENT_INVALID_ARGS,
                            error_message=f"spawn_agent 参数无效: {exc}",
                        )
                    else:
                        spawn_result = await invoke_spawn(
                            request=request,
                            serena_client=serena_client,
                            parent_depth=depth,
                            parent_deadline=deadline,
                            run_loop_fn=self._run_react_loop,
                        )
                        spawn_count += 1
                        # 子 Agent 成功结论也写入 InsightCache
                        if spawn_result.ok:
                            await self._persist_insights(
                                question=request.task,
                                answer=spawn_result.answer,
                                confidence=spawn_result.confidence,
                            )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": "spawn_agent",
                        "content": _safe_json_dumps(spawn_result.to_tool_content()),
                    }
                )

            # 处理认知确认调用
            for tc in acknowledge_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    tc_id = tc.get("id") or str(uuid.uuid4())
                    raw_args = fn.get("arguments") or tc.get("arguments")
                else:
                    fn = getattr(tc, "function", None)
                    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())
                    raw_args = getattr(fn, "arguments", None)
                args = _parse_tool_call_args(raw_args)
                paths = (args or {}).get("paths", [])
                if acknowledge_cb and paths:
                    try:
                        await acknowledge_cb(paths)
                        ack_result = f"已确认 {len(paths)} 个文件的认知状态"
                    except Exception as e:
                        ack_result = f"认知确认失败: {e}"
                else:
                    ack_result = "无需确认（无变更文件或回调未配置）"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": "acknowledge_changes",
                        "content": ack_result,
                    }
                )

            # deliver 到来，提取结论并正常终止
            if deliver_call is not None:
                if isinstance(deliver_call, dict):
                    fn = deliver_call.get("function") or {}
                    raw_args = fn.get("arguments") or deliver_call.get("arguments")
                else:
                    fn = getattr(deliver_call, "function", None)
                    raw_args = getattr(fn, "arguments", None)
                args = _parse_tool_call_args(raw_args)
                if args is None:
                    logger.warning("deliver 调用参数解析失败")
                    return "__REACT_DELIVER_EMPTY__", None
                answer = args.get("answer")
                if not answer:
                    logger.warning("deliver 调用缺少 answer 参数")
                    return "__REACT_DELIVER_EMPTY__", None
                confidence = args.get("confidence")
                elapsed = time.monotonic() - start
                logger.debug(f"ReAct 循环收到 deliver，正常终止（耗时 {elapsed:.1f}s）")
                return str(answer), str(confidence) if confidence else None

            # 本轮仅有 Serena 工具调用，重置无工具计数，继续下一轮
            no_tool_retries = 0

    async def _invoke_tool(
        self, tool_call: dict[str, Any], serena_client: SerenaClient
    ) -> dict[str, Any]:
        """执行单个工具调用,返回格式化的 tool 消息。"""
        # 提取 tool_call 字段(兼容 dict 和 object 两种格式)
        if isinstance(tool_call, dict):
            tool_call_id = tool_call.get("id") or str(uuid.uuid4())
            function = tool_call.get("function") or {}
            tool_name = function.get("name") or tool_call.get("name")
            raw_args = function.get("arguments") or tool_call.get("arguments")
        else:
            tool_call_id = getattr(tool_call, "id", None) or str(uuid.uuid4())
            function = getattr(tool_call, "function", {})
            tool_name = getattr(function, "name", None)
            raw_args = getattr(function, "arguments", None)

        if not tool_name:
            return {
                "tool_call_id": tool_call_id,
                "name": "unknown",
                "content": "工具调用缺少 name 字段",
            }

        args = _parse_tool_call_args(raw_args)

        # args 为 None 说明 JSON 解析失败（参数不完整或格式错误）
        # 返回错误 tool result，让 LLM 感知并重新构造正确的调用
        if args is None:
            logger.warning(f"工具参数解析失败: {tool_name}, raw_args={raw_args!r:.100}")
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"工具参数解析失败: 无效的 JSON。请重新调用 {tool_name} 并确保参数格式正确。",
            }

        try:
            result = await serena_client.call(tool_name, args)
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": _safe_json_dumps(result),
            }
        except SerenaClientError as e:
            logger.warning(f"工具调用失败: {tool_name}: {e}")
            return {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": f"工具调用失败: {e}",
            }

    # ============================================================================
    # 公开接口
    # ============================================================================

    async def query(
        self,
        question: str,
        session_id: str | None = None,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> QueryResponse:
        """执行自然语言查询。

        Args:
            question: 用户问题
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            结构化的查询响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        sid = session_id or str(uuid.uuid4())
        messages = list(self._sessions.get(sid, []))

        # 新会话时注入 system prompt
        if not messages:
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            messages = [{"role": "system", "content": system_content}]

        messages.append({"role": "user", "content": question})
        answer, confidence = await self._run_react_loop(messages, serena_client, acknowledge_cb)
        self._sessions[sid] = messages
        await self._persist_insights(question=question, answer=answer, confidence=confidence)

        return _parse_query_response(answer, sid)

    async def impact(
        self,
        target: str,
        change_type: str,
        session_id: str | None = None,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> ImpactResponse:
        """执行变更影响分析。

        Args:
            target: 分析目标(符号名或文件路径)
            change_type: 变更类型(modify/rename/delete/add_field/change_signature)
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            结构化的影响分析响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        sid = session_id or str(uuid.uuid4())
        messages = list(self._sessions.get(sid, []))

        if not messages:
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            messages = [{"role": "system", "content": system_content}]

        # 构造影响分析专用提示词
        prompt = "\n".join(
            [
                f"请分析对 `{target}` 进行 `{change_type}` 变更的影响边界。",
                "",
                "请使用 Serena 工具查找所有引用,然后以 JSON 格式输出:",
                "{",
                '  "impact_chain": [...],  // 影响链路中的 ReferenceEdge 列表',
                '  "boundary": [...],      // 影响边界的 SymbolRef 列表',
                '  "risks": [...],         // 风险提示字符串列表',
                '  "unknowns": [...]       // 不确定项字符串列表',
                "}",
            ]
        )

        messages.append({"role": "user", "content": prompt})
        answer, confidence = await self._run_react_loop(messages, serena_client, acknowledge_cb)
        self._sessions[sid] = messages
        await self._persist_insights(question=prompt, answer=answer, confidence=confidence)

        return _parse_impact_response(answer, sid)
