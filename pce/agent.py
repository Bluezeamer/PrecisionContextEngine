"""PCE Agent — 手工实现的 ReAct (Reasoning + Acting) 循环。

基于 litellm 的 tool_call 流程驱动 Serena 工具调用,实现:
- 自然语言代码库查询 (pce_query)
- 变更影响边界分析 (pce_impact)

会话管理:
- in-memory 存储 `_sessions: dict[str, list[dict]]`
- 每次推理前执行滑动窗口截断(保留最近 WINDOW_MAX_PAIRS 轮)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from .models import ImpactResponse, QueryResponse, ReferenceEdge, SymbolRef
from .serena_client import SerenaClient, SerenaClientError

logger = logging.getLogger(__name__)

MAX_STEPS = 10
WINDOW_MAX_PAIRS = 8
MODEL = os.getenv("PCE_MODEL", "step-3.5-flash")

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

输出要求:
- 回答简洁精准,优先给出文件路径和行号定位
- 影响分析时列出所有直接引用点,不要遗漏\
"""


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
        return message.model_dump()
    if isinstance(message, dict):
        return message
    return {"role": "assistant", "content": str(message)}


def _extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """从 message 中提取 tool_calls 列表。"""
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _parse_tool_call_args(raw_args: Any) -> dict[str, Any]:
    """解析 tool_call 的 arguments 字段。"""
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            result = json.loads(raw_args)
            return result if isinstance(result, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}
    return {}


def _apply_sliding_window(
    messages: list[dict[str, Any]], max_pairs: int
) -> list[dict[str, Any]]:
    """截断 messages 列表,保留 system prompt + 最近 N 轮对话。

    一轮对话约 3 条消息: user + assistant + tool(s)
    超出时从最早的非 system 消息开始丢弃。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    max_non_system = max_pairs * 3
    if len(non_system) > max_non_system:
        non_system = non_system[-max_non_system:]
        logger.debug(f"滑动窗口截断: 保留最后 {max_non_system} 条非 system 消息")

    return system_msgs + non_system


def _best_effort_answer(messages: list[dict[str, Any]]) -> str:
    """超出最大步数时,从历史消息中提取最佳答案。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return "未能在限定步数内得到结论,请缩小问题范围或重试。"


# ============================================================================
# 响应解析
# ============================================================================


def _parse_query_response(content: str, session_id: str) -> QueryResponse:
    """将 Agent 输出解析为 QueryResponse。"""
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
        max_steps: int = MAX_STEPS,
        window_max_pairs: int = WINDOW_MAX_PAIRS,
    ) -> None:
        self._model = model or MODEL
        self._max_steps = max_steps
        self._window_max_pairs = window_max_pairs
        # in-memory 会话存储,进程重启后丢失(MVP 范围内可接受)
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    async def _read_pce_file(self, path: Path) -> str:
        """读取 .pce 目录下的文件内容,文件不存在时返回占位文本。"""
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return f"(未找到 {path.name},请先执行 pce_init 初始化索引)"
        except Exception as e:
            logger.warning(f"读取文件失败: {path}: {e}")
            return f"(读取 {path.name} 失败)"

    async def _build_system_prompt(self, memory_root: Path | None) -> str:
        """构建 system prompt,注入 structure.md 和 annotations.md 内容。"""
        root = memory_root or Path.cwd()
        pce_dir = root / ".pce"

        structure_md = await self._read_pce_file(pce_dir / "structure.md")
        annotations_md = await self._read_pce_file(pce_dir / "annotations.md")

        return "\n".join([
            SYSTEM_PROMPT_HEADER,
            "",
            "## 项目结构 (structure.md)",
            structure_md.strip(),
            "",
            "## 语义注解 (annotations.md)",
            annotations_md.strip(),
        ])

    async def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """调用 litellm.completion(在线程池中执行,设置 60 秒超时)。"""
        return await asyncio.wait_for(
            asyncio.to_thread(
                litellm.completion,
                model=self._model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.2,
            ),
            timeout=60.0,
        )

    async def _run_react_loop(
        self,
        messages: list[dict[str, Any]],
        serena_client: SerenaClient,
    ) -> str:
        """执行 ReAct 循环直到 LLM 输出结论或达到步数上限。

        Returns:
            LLM 最终输出的内容字符串
        """
        for step in range(self._max_steps):
            # 每次 LLM 调用前执行滑动窗口截断
            windowed = _apply_sliding_window(messages, self._window_max_pairs)

            response = await self._completion(windowed, serena_client.tools_schema)
            message = _extract_message(response)
            if "role" not in message:
                message["role"] = "assistant"

            # 追加 assistant 消息保持对话连续性
            messages.append(message)

            tool_calls = _extract_tool_calls(message)

            # 无 tool_call → LLM 认为信息已足够,输出结论
            if not tool_calls:
                content = message.get("content") or ""
                logger.debug(f"ReAct 循环结束于第 {step + 1} 步")
                return str(content)

            # 并行执行所有 tool_calls
            tool_results = await asyncio.gather(
                *[self._invoke_tool(tc, serena_client) for tc in tool_calls]
            )

            # 将工具结果追加回 messages
            for tool_result_msg in tool_results:
                messages.append({"role": "tool", **tool_result_msg})

        logger.warning(f"ReAct 循环达到最大步数 {self._max_steps},返回最佳答案")
        return _best_effort_answer(messages)

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
    ) -> QueryResponse:
        """执行自然语言查询。

        Args:
            question: 用户问题
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient

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
        answer = await self._run_react_loop(messages, serena_client)
        self._sessions[sid] = messages

        return _parse_query_response(answer, sid)

    async def impact(
        self,
        target: str,
        change_type: str,
        session_id: str | None = None,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
    ) -> ImpactResponse:
        """执行变更影响分析。

        Args:
            target: 分析目标(符号名或文件路径)
            change_type: 变更类型(modify/rename/delete/add_field/change_signature)
            session_id: 会话 ID,不传时自动创建新会话
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient

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
        prompt = "\n".join([
            f"请分析对 `{target}` 进行 `{change_type}` 变更的影响边界。",
            "",
            "请使用 Serena 工具查找所有引用,然后以 JSON 格式输出:",
            "{",
            '  "impact_chain": [...],  // 影响链路中的 ReferenceEdge 列表',
            '  "boundary": [...],      // 影响边界的 SymbolRef 列表',
            '  "risks": [...],         // 风险提示字符串列表',
            '  "unknowns": [...]       // 不确定项字符串列表',
            "}",
        ])

        messages.append({"role": "user", "content": prompt})
        answer = await self._run_react_loop(messages, serena_client)
        self._sessions[sid] = messages

        return _parse_impact_response(answer, sid)
