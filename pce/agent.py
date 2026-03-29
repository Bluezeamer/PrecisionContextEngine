"""PCE Agent — 手工实现的 ReAct (Reasoning + Acting) 循环。

基于 litellm 的 tool_call 流程驱动 Serena 工具调用,实现:
- 自然语言代码库查询 (pce_query)
- 变更影响边界分析 (pce_impact)

运行特性:
- 每次调用从 Memory 快照与 InsightCache 注入重新起步（无状态）
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
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from ._env import get_system_prompt_soft_limit
from .agent_runtime.contracts import (
    MAX_SPAWNS_PER_LOOP,
    SPAWN_AGENT_TOOL,
    SpawnErrorCode,
    SpawnRequest,
    SpawnResult,
    SpawnStatus,
)
from .agent_runtime.spawner import invoke_spawn
from .base_agent import (
    REACT_DELIVER_EMPTY,
    REACT_INPUT_TOO_LARGE,
    BaseReActAgent,
    DeliverDecision,
    LLMCompletionError,
    LoopState,
    _extract_message,
    _extract_tool_call_args,
    _get_tool_call_id,
    _get_tool_name,
    _safe_json_dumps,
)
from .file_discovery import is_visible_to_agent
from .insight_cache import InsightCache
from .models import ImpactResponse, InsightConfidence, QueryResponse
from .prompt_guard import build_prompt_budget, estimate_input_tokens, fit_text_to_budget
from .serena_client import SerenaClient, SerenaClientError

# 认知确认回调：接受文件路径列表，执行暂存区标记
AcknowledgeCallback = Callable[[list[str]], Awaitable[None]]

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

**工作方法论**:
- 先判断任务目标，再决定证据深度；不要一上来就追求“全都找全”。
- 若任务是“先找到入口/定义/主链附近”，优先收敛到足够支持上层继续工作的最小证据集。
- 若任务是“目标已明确后的影响分析”，优先确认直接调用点、直接消费者和主要传播链，再决定是否继续补远端弱链路。
- 证据强度高于叙述完整度；宁可少写，也不要为了填满某个 section 去补推断。
- 直接证据（直接调用、直接解构、直接传参、直接响应构造）优先于弱证据（UI 展示、派生统计、附属导出、按钮状态）。
- 若只能确认到文件层，就写 `path`；不要为了“看起来完整”去猜测 `path:line`。

**query / impact 边界**:
- query 的目标是把上层 Agent 快速送到目标附近：入口、定义点、主调用链、关键文件、关键符号。
- impact 的目标是围绕已知 target 给出高价值影响边界：直接调用点、直接消费者、主要传播链、风险与修改顺序。
- 不要把 query 做成穷尽式传播链分析；也不要在 impact 中替代“找目标”这一步。

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
上层 Agent 可在 query 中补充字段要求或格式细节。PCE 的主输出协议为**结构化 Markdown**。
无论是 query 还是 impact,最终都必须调用 `deliver(answer=..., confidence=...)`。
其中 `deliver(answer=...)` 的 `answer` 必须是**结构清晰的 Markdown 文本**,不得输出 JSON。

query 任务默认 Markdown 结构:
## 结论
...

## 关键证据
- `path:line` ...

## 相关符号
- `symbol` — `kind` — `path:line`

## 相关文件
- `path`

## 不确定项
- ...

impact 任务默认 Markdown 结构:
## 直接调用点
- `path:line` — 为什么这里会受影响

## 数据契约/返回值消费者
- `path:line` — 哪个返回值、字段、解析点或消费者会直接受影响

## 间接传播链
- `path:line` — 下游传递、UI 触发或后续消费链路

## 边界符号
- `symbol` — `kind` — `path:line`

## 风险
- ...

## 不确定项
- ...

## 建议修改顺序
1. ...
2. ...

输出要求:
- 回答简洁精准,优先给出文件路径和行号定位
- 使用固定标题,不要改写标题名称
- impact 必须继续追踪 downstream 传播链,不要只停留在符号导入层
- 只有工具证据直接确认过的定位才能写成 `path:line`; 若只确认到文件层,写 `path`，不要编造行号
- 每条 bullet 只能绑定一个定位；说明必须只描述该定位所在文件中的直接行为，跨文件关系必须拆成多条
- 在 deliver 前，应回读或检索你准备写出的每一个 `path:line`；无法复核的条目必须去掉行号
- `直接调用点` 只写直接调用、直接解析、直接构造、直接字段消费的位置；按钮触发、页面跳转、后续流程归入“间接传播链”
- `数据契约/返回值消费者` 只写返回值、响应字段、表单字段、序列化/反序列化等直接契约面
- 不要用上游文件的位置去承载下游文件的行为；如果要描述前端接收，就必须列前端文件自己的定位
- 如果某个 section 证据不足，可以少写；不要机械地把每个 section 都填满
- 若某项无法确认,写入“不确定项”,不要编造\
"""

IMPACT_STRATEGY_SYSTEM_PROMPT = """\
你是 PCE 的 impact 前置策略分析器。

职责：
1. 仅根据任务描述判断当前 impact 更偏向 symbol-first、dataflow-first 还是 mixed。
2. 给出“优先取证顺序”和“工具使用倾向”，用于指导后续主 Agent。
3. 这是策略画像，不是最终答案；不要分析代码事实，不要假装已经验证过任何结论。

要求：
- 输出简短、克制、可直接嵌入提示词。
- 不输出 JSON。
- 不要写不存在的文件、行号、符号。
- 工具使用建议只能表达“优先级和倾向”，不能写成硬性禁令。
- 若任务语义同时包含符号边界与字段传播，请优先判定为 mixed。
"""

# deliver 是一个虚拟工具,不转发给 Serena,由循环内部拦截处理
# agent 调用它表示主动声明"任务完成",是唯一的正常终止信号
DELIVER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "deliver",
        "description": (
            "提交最终结论并结束当前任务。完成推理后必须调用此工具,不得直接以文字回答。"
            "answer 必须是结构化 Markdown 文本,不得输出 JSON 或 Markdown 代码块。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "最终结论内容，必须是结构化 Markdown 文本",
                },
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
# system prompt 动态注入块软上限（token 数），运行时从 env 读取。
# 仅针对可压缩块（structure.md + annotations/index.md + InsightCache），
# 不含 SYSTEM_PROMPT_HEADER 和工具 schema。
# 计算方式：PCE_CONTEXT_WINDOW / 10，默认 200k / 10 = 20000。
_SYSTEM_PROMPT_SOFT_LIMIT: int = get_system_prompt_soft_limit()
_SYSTEM_PROMPT_TRUNCATED_NOTICE = (
    "\n\n(以上内容因 system prompt token 预算限制已被截断，如需完整信息请通过工具按需读取)"
)
_SYSTEM_PROMPT_PLACEHOLDER = "(内容因 system prompt token 超限未注入，请通过工具按需读取)"
_INSIGHT_MAX_CONTENT_CHARS = 1200  # 单条 insight content 最大字符数
# 从文本中提取形如 pce/agent.py 或 ./src/foo.py:123 的路径
_PATH_RE = re.compile(r"(?<![\w./-])(?:\./)?([A-Za-z0-9_./-]+\.[A-Za-z0-9_+\-]+)(?::\d+)?")


# ---------------------------------------------------------------------------
# 请求追踪（per-request，无状态）
# ---------------------------------------------------------------------------

# 每次 pce_query/pce_impact 在入口设置，通过 asyncio context 自动传播给 spawn 子 Agent
_req_id_var: ContextVar[str] = ContextVar("pce_req_id", default="")

# Serena 工具名 → 最能代表查询意图的主要参数名（用于 INFO 日志预览）
_PRIMARY_ARG: dict[str, str] = {
    "find_symbol": "name_path_pattern",
    "find_referencing_symbols": "name_path",
    "get_symbols_overview": "relative_path",
    "search_for_pattern": "substring_pattern",
    "list_dir": "relative_path",
    "find_file": "file_mask",
    "read_file": "relative_path",
}


def _key_arg_preview(tool_name: str, args: dict[str, Any] | None) -> str:
    """提取工具主要参数预览，返回 'tool(key=val)' 格式，用于 INFO 日志。"""
    if not args:
        return tool_name

    primary_key = _PRIMARY_ARG.get(tool_name)
    display_key: str | None = None
    value: str | None = None

    # 优先取 _PRIMARY_ARG 对应的参数
    if primary_key:
        candidate = args.get(primary_key)
        if isinstance(candidate, str) and candidate:
            display_key = primary_key
            value = candidate

    # fallback：取第一个非空字符串值（不知道 key 名，只显示值）
    if value is None:
        for v in args.values():
            if isinstance(v, str) and v:
                value = v
                break

    if value is None:
        return tool_name

    truncated = value[:57].rstrip() + "..." if len(value) > 60 else value
    return (
        f"{tool_name}({display_key}={truncated!r})"
        if display_key
        else f"{tool_name}({truncated!r})"
    )


# ---------------------------------------------------------------------------
# 可选 JSONL Trace 文件写入（PCE_TRACE_DIR 设置时生效）
# ---------------------------------------------------------------------------


class _TraceWriter:
    """per-request JSONL 追踪写入器。

    每条消息（system/user/assistant/tool/deliver）异步追加到独立文件。
    所有 I/O 均 best-effort：失败不影响主流程。
    """

    _MAX_FILES: int = 50  # 保留最近 N 个 trace 文件

    def __init__(self, req_id: str, trace_dir: Path) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = trace_dir / f"{ts}_{req_id}.jsonl"

    @classmethod
    def from_env(cls, req_id: str) -> _TraceWriter | None:
        """若 PCE_TRACE_DIR 已配置则创建实例，否则返回 None。"""
        raw = os.getenv("PCE_TRACE_DIR", "").strip()
        if not raw:
            return None
        trace_dir = Path(raw).expanduser()
        try:
            trace_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None
        return cls(req_id, trace_dir)

    async def write(self, record: dict[str, Any]) -> None:
        """将 record 追加为一行 JSON 到 trace 文件。"""
        try:
            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            async with aiofiles.open(self._path, "a", encoding="utf-8") as f:
                await f.write(line)
        except Exception:
            pass

    async def rotate(self) -> None:
        """超出上限时删除最旧的 trace 文件（LRU）。"""
        try:
            files = sorted(
                self._path.parent.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
            )
            for old_file in files[: -self._MAX_FILES]:
                old_file.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================================
# 辅助函数
# ============================================================================


def _try_parse_json(value: Any) -> Any:
    """尝试将字符串解析为 JSON,失败时原样返回。"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass
    return value


# ============================================================================
# 响应解析辅助
# ============================================================================


def _norm_text(value: Any) -> str | None:
    """将任意输入归一化为非空字符串。"""
    if isinstance(value, Path):
        text = value.as_posix()
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    return text or None


def _norm_line(value: Any) -> int | None:
    """将任意输入归一化为合法行号（>= 1）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed >= 1 else None
    return None


def _extract_lightweight_symbol(item: dict[str, Any]) -> dict[str, Any] | None:
    """从 LLM 输出中提取轻量符号信息（不要求 UUID）。"""
    sym: dict[str, Any] = {}

    name = _norm_text(item.get("name"))
    name_path = _norm_text(item.get("name_path"))
    if name:
        sym["name"] = name
    elif name_path:
        derived = name_path.split("/")[-1]
        if not derived:
            return None  # name_path 尾部为空，无法提取有效名称
        sym["name"] = derived
    else:
        return None  # 至少需要名字

    kind = _norm_text(item.get("kind"))
    if kind:
        sym["kind"] = kind
    file_path = _norm_text(item.get("file_path") or item.get("file"))
    if file_path:
        sym["file_path"] = file_path
    if name_path:
        sym["name_path"] = name_path

    line_range = item.get("line_range")
    line_start = _norm_line(item.get("line_start")) or _norm_line(item.get("line"))
    if line_start is None and isinstance(line_range, (list, tuple)) and line_range:
        line_start = _norm_line(line_range[0])
    if line_start is not None:
        sym["line_start"] = line_start

    line_end = _norm_line(item.get("line_end"))
    if line_end is None and isinstance(line_range, (list, tuple)) and len(line_range) >= 2:
        line_end = _norm_line(line_range[1])
    if line_end is None and line_start is not None:
        line_end = line_start
    if line_end is not None:
        sym["line_end"] = max(line_end, line_start or line_end)

    signature = _norm_text(item.get("signature"))
    if signature:
        sym["signature"] = signature
    return sym


def _extract_lightweight_edge(item: dict[str, Any]) -> dict[str, Any] | None:
    """从 LLM 输出中提取 impact_chain 的轻量引用边信息。"""
    edge: dict[str, Any] = {}

    file_path = _norm_text(item.get("file") or item.get("file_path"))
    if file_path:
        edge["file"] = file_path

    line = _norm_line(item.get("line")) or _norm_line(item.get("line_start"))
    if line is not None:
        edge["line"] = line

    ref_sym = _norm_text(
        item.get("referencing_symbol") or item.get("from_symbol") or item.get("source_symbol")
    )
    if ref_sym:
        edge["referencing_symbol"] = ref_sym

    snippet = _norm_text(item.get("snippet") or item.get("evidence"))
    if snippet:
        edge["snippet"] = snippet

    relation = _norm_text(item.get("relation"))
    if relation:
        edge["relation"] = relation

    # 至少需要 file 字段才算有效引用点
    return edge if "file" in edge else None


def _looks_like_markdown_sections(content: str) -> bool:
    """判断文本是否已经具备分段 Markdown 结构。"""
    return "## " in content


_SECTION_TITLE_RE = re.compile(r"^##\s+(.+?)\s*$")
_LOCATION_TOKEN_RE = re.compile(r"`((?:\./)?[A-Za-z0-9_./-]+\.[A-Za-z0-9_+\-]+)(?::(\d+))?`")
_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}")
_COMMENT_PREFIXES = ("#", "//", "/*", "*", "<!--")
_IMPORT_PREFIXES = ("from ", "import ")
_GENERIC_LOCATION_WORDS = {
    "backend",
    "frontend",
    "layer",
    "order",
    "result",
    "value",
    "data",
    "path",
    "line",
    "jsonresponse",
    "function",
    "symbol",
    "change",
    "signature",
    "target",
}
_GENERIC_STRATEGY_KEYS = {
    "api",
    "app",
    "backend",
    "frontend",
    "modify",
    "change",
    "signature",
    "delete",
    "rename",
    "field",
    "layer",
    "order",
    "file",
    "target",
    "function",
    "response",
    "request",
    "return",
    "data",
}


def _split_markdown_sections(content: str) -> list[tuple[str, list[str]]]:
    """按二级标题切分 Markdown，保留每个 section 的原始行。"""
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        matched = _SECTION_TITLE_RE.match(line)
        if matched:
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = matched.group(1).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, current_lines))
    return sections


def _extract_location_keywords(bullet: str) -> list[str]:
    """从 bullet 文本中提取用于纠偏的关键词。"""
    text = _LOCATION_TOKEN_RE.sub(" ", bullet)
    backticked = re.findall(r"`([^`]+)`", text)
    lowered = text.lower()
    is_response_like = (
        "响应" in text or "返回" in text or "response" in lowered or "return" in lowered
    )
    keywords: list[str] = []
    seen: set[str] = set()

    def add_token(token: str) -> None:
        normalized = token.strip().strip("`'\"").lower()
        if len(normalized) < 3 or normalized in seen or normalized in _GENERIC_LOCATION_WORDS:
            return
        seen.add(normalized)
        keywords.append(normalized)

    for token in backticked:
        for piece in _KEYWORD_TOKEN_RE.findall(token):
            normalized_piece = piece.strip().strip("`'\"").lower()
            if (
                is_response_like
                and re.fullmatch(r"[a-z_][a-z0-9_]*", normalized_piece)
                and lowered.count(normalized_piece) < 2
            ):
                continue
            add_token(piece)
        if is_response_like:
            for quoted_key in re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*:', token):
                add_token(f'"{quoted_key}":')
                add_token(f"'{quoted_key}':")
    for piece in _KEYWORD_TOKEN_RE.findall(text):
        add_token(piece)

    if "接收" in text or "receive" in lowered or "consumer" in lowered:
        for hint in ("data.data", "res.json", ".value =", "await fetch", "await res.json"):
            add_token(hint)
    if is_response_like:
        for hint in ("jsonresponse(", '"data":', "'data':", "data = {", "return {"):
            add_token(hint)
        for token in backticked:
            normalized = token.strip().strip("`'\"").lower()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", normalized):
                add_token(f'"{normalized}":')
                add_token(f"'{normalized}':")
    if "发送" in text or "传递" in text or "submit" in lowered or "post" in lowered:
        for hint in ("json.stringify", "formdata", "fd.append", "await fetch", "/api/"):
            add_token(hint)
    if "调用" in text or "call" in lowered:
        for hint in ("(", "return", "="):
            add_token(hint)
    return keywords


def _line_looks_like_import(lines: list[str], line_no: int) -> bool:
    """识别单行导入或多行导入块中的一行。"""
    if line_no < 1 or line_no > len(lines):
        return False
    text = lines[line_no - 1].strip().lower()
    if text.startswith(_IMPORT_PREFIXES):
        return True
    for idx in range(max(1, line_no - 12), line_no):
        prev = lines[idx - 1].strip().lower()
        if prev.startswith(_IMPORT_PREFIXES):
            return True
        if prev.endswith(" import (") or prev.endswith(" import("):
            return True
    return False


def _score_line_window(lines: list[str], line_no: int, keywords: list[str]) -> int:
    """对某行附近窗口进行关键词匹配打分。"""
    if line_no < 1 or line_no > len(lines):
        return -1
    exact = lines[line_no - 1].lower()
    context_lines = []
    for idx in range(max(1, line_no - 2), min(len(lines), line_no + 1) + 1):
        if idx == line_no:
            continue
        context_lines.append(lines[idx - 1])
    window = "\n".join(context_lines).lower()

    def keyword_weight(keyword: str) -> int:
        if keyword.endswith('":') or keyword.endswith("':"):
            return 4
        if re.fullmatch(r"[a-z_][a-z0-9_]*", keyword):
            return 2
        return 1

    def keyword_present(keyword: str, text: str) -> bool:
        if re.fullmatch(r"[a-z_][a-z0-9_]*", keyword):
            return re.search(rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])", text) is not None
        return keyword in text

    exact_hits = sum(keyword_weight(kw) for kw in keywords if keyword_present(kw, exact))
    window_hits = sum(keyword_weight(kw) for kw in keywords if keyword_present(kw, window))
    penalty = 0
    if _line_looks_like_import(lines, line_no):
        penalty -= 3
    return exact_hits * 2 + window_hits + penalty


def _line_looks_unreliable(lines: list[str], line_no: int) -> bool:
    """判断某行是否像是注释/空行/装饰行，适合作为纠偏候选。"""
    if line_no < 1 or line_no > len(lines):
        return True
    text = lines[line_no - 1].strip()
    if not text:
        return True
    return text.startswith(_COMMENT_PREFIXES)


def _find_better_line_in_file(
    lines: list[str], claimed_line: int, keywords: list[str]
) -> int | None:
    """在文件内为某个 bullet 找更可信的行号；找不到则返回 None。"""
    current_score = _score_line_window(lines, claimed_line, keywords)
    current_unreliable = _line_looks_unreliable(lines, claimed_line)
    has_structural_keywords = any(re.search(r'["\'=:(){}.]', kw) for kw in keywords)
    should_search_beyond_claimed = (
        _line_looks_like_import(lines, claimed_line) and has_structural_keywords
    )

    best_line = claimed_line
    best_score = current_score

    for candidate in range(max(1, claimed_line - 3), min(len(lines), claimed_line + 3) + 1):
        score = _score_line_window(lines, candidate, keywords)
        if score > best_score:
            best_score = score
            best_line = candidate

    if best_line != claimed_line and best_score > current_score:
        return best_line

    if not current_unreliable and current_score > 0 and not should_search_beyond_claimed:
        return claimed_line

    search_best_line = claimed_line
    search_best_score = current_score
    for idx in range(1, len(lines) + 1):
        score = _score_line_window(lines, idx, keywords)
        if score > search_best_score:
            search_best_score = score
            search_best_line = idx

    if search_best_score > max(current_score, 0):
        return search_best_line
    return None if current_unreliable or current_score <= 0 else claimed_line


def _find_best_line_by_keywords(lines: list[str], keywords: list[str]) -> int | None:
    """在没有明确行号时，仅凭关键词为 bare path 选择最可信的行号。"""
    if not keywords:
        return None

    best_line: int | None = None
    best_score = 0
    for idx in range(1, len(lines) + 1):
        score = _score_line_window(lines, idx, keywords)
        if score > best_score:
            best_score = score
            best_line = idx
    return best_line if best_score > 0 else None


def _normalize_location_token(
    file_token: str,
    line_token: str | None,
    *,
    project_root: Path | None,
    bullet: str,
) -> str:
    """复核并修正 Markdown 中的 `path:line` 定位，失败时降级为 `path`。"""
    file_path = file_token[2:] if file_token.startswith("./") else file_token
    if project_root is None:
        return f"`{file_path}:{line_token}`" if line_token else f"`{file_path}`"

    abs_path = project_root / file_path
    if not abs_path.exists() or not abs_path.is_file():
        return f"`{file_path}`"

    try:
        lines = abs_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        try:
            lines = abs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return f"`{file_path}`"
    except Exception:
        return f"`{file_path}`"

    keywords = _extract_location_keywords(bullet)
    if not line_token:
        inferred_line = _find_best_line_by_keywords(lines, keywords)
        return f"`{file_path}:{inferred_line}`" if inferred_line is not None else f"`{file_path}`"

    claimed_line = _norm_line(line_token)
    if claimed_line is None:
        return f"`{file_path}`"

    better_line = _find_better_line_in_file(lines, claimed_line, keywords)
    if better_line is None:
        return f"`{file_path}`"
    return f"`{file_path}:{better_line}`"


def _post_validate_markdown_locations(
    content: str, *, project_root: str | Path | None = None
) -> str:
    """对 Markdown 做轻量复核与纠偏，主要校验 `path:line`。"""
    root_path = Path(project_root).resolve() if project_root else None
    rebuilt_sections: list[str] = []

    for title, lines in _split_markdown_sections(content):
        rebuilt_sections.append(f"## {title}")
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("- "):
                updated = _LOCATION_TOKEN_RE.sub(
                    lambda m: _normalize_location_token(
                        m.group(1), m.group(2), project_root=root_path, bullet=line
                    ),
                    line,
                )
                rebuilt_sections.append(updated)
            else:
                rebuilt_sections.append(line)
        rebuilt_sections.append("")

    return "\n".join(rebuilt_sections).strip()


def _should_filter_contract_bullet(title: str, line: str, *, target: str) -> bool:
    """对字段传播类任务做 very light 去歧义，过滤同名本地表单/派生状态。"""
    lowered_target = target.lower()
    is_contract_task = (
        "字段" in target
        or "返回字段" in target
        or "请求字段" in target
        or "response field" in lowered_target
        or "request field" in lowered_target
        or "/api/" in target
    )
    if not is_contract_task:
        return False
    if title not in {"数据契约/返回值消费者", "间接传播链"}:
        return False

    lowered = line.lower()
    if "form.value." in lowered or "form." in lowered or "本地表单" in line:
        return True
    if "派生状态" in line or "computed" in lowered or "derived" in lowered:
        return True
    return False


def _looks_like_direct_consumer_bullet(line: str) -> bool:
    """判断 bullet 是否更像直接消费者/直接契约面。"""
    lowered = line.lower()
    semantic_markers = (
        "直接消费",
        "直接接收",
        "接收返回值",
        "接收响应",
        "解构返回值",
        "承接返回值",
        "直接赋值",
        "直接存入",
        "作为参数",
        "作为输入",
        "传递给下游",
        "传入下游",
        "解析为",
        "序列化边界",
        "反序列化边界",
        "构造请求",
        "构造响应",
        "写入请求",
        "写入响应",
        "直接消费者",
        "direct consumer",
        "receive response",
        "receive return value",
        "deserialize",
        "serialize",
        "request construction",
        "response construction",
    )
    return any(marker in lowered for marker in semantic_markers)


def _looks_like_weak_indirect_bullet(line: str) -> bool:
    """判断 bullet 是否更像弱传播链/派生展示链。"""
    lowered = line.lower()
    semantic_markers = (
        "用于展示",
        "用于显示",
        "用于渲染",
        "用于预览",
        "用于下载",
        "用于导出",
        "用于统计",
        "用于提示",
        "用于界面",
        "用于按钮状态",
        "用于样式状态",
        "派生字段",
        "派生信息",
        "附属信息",
        "元信息",
        "弱传播链",
        "indirect display",
        "used for display",
        "used for rendering",
        "used for preview",
        "used for export",
        "used for statistics",
        "metadata",
        "derived info",
    )
    return any(marker in lowered for marker in semantic_markers)


def _sanitize_risk_line(line: str) -> str:
    """风险段落只保留自然语言，不保留精确行号。"""
    line = re.sub(
        r"`((?:\./)?[A-Za-z0-9_./-]+\.[A-Za-z0-9_+\-]+):(\d+)`",
        lambda m: f"`{m.group(1)}`",
        line,
    )
    line = re.sub(r"((?:\./)?[A-Za-z0-9_./-]+\.[A-Za-z0-9_+\-]+):(\d+)", r"\1", line)
    return line


def _post_filter_impact_markdown(content: str, *, target: str) -> str:
    """对 impact Markdown 做轻量语义去歧义过滤。"""
    sections = _split_markdown_sections(content)
    contract_lines: list[str] = []
    indirect_lines: list[str] = []
    passthrough_sections: list[tuple[str, list[str]]] = []

    for title, lines in sections:
        if title not in {"数据契约/返回值消费者", "间接传播链"}:
            if title == "风险":
                cleaned = [
                    _sanitize_risk_line(line) if line.lstrip().startswith("- ") else line
                    for line in lines
                ]
                passthrough_sections.append((title, cleaned))
            else:
                passthrough_sections.append((title, list(lines)))
            continue

        passthrough_sections.append((title, []))

        for line in lines:
            stripped = line.lstrip()
            if not stripped.startswith("- "):
                continue
            if _should_filter_contract_bullet(title, line, target=target):
                continue
            if title == "数据契约/返回值消费者" and _looks_like_weak_indirect_bullet(line):
                indirect_lines.append(line)
                continue
            if title == "间接传播链" and _looks_like_direct_consumer_bullet(line):
                contract_lines.append(line)
                continue
            if title == "数据契约/返回值消费者":
                contract_lines.append(line)
            else:
                indirect_lines.append(line)

    rebuilt_sections: list[str] = []
    for title, lines in passthrough_sections:
        rebuilt_sections.append(f"## {title}")
        if title == "数据契约/返回值消费者":
            lines = contract_lines or ["- （未结构化提供）"]
        elif title == "间接传播链":
            lines = indirect_lines or ["- （未结构化提供）"]
        rebuilt_sections.extend(lines)
        rebuilt_sections.append("")

    seen_titles = {title for title, _ in passthrough_sections}
    if "数据契约/返回值消费者" not in seen_titles:
        rebuilt_sections.extend(
            ["## 数据契约/返回值消费者", *(contract_lines or ["- （未结构化提供）"]), ""]
        )
    if "间接传播链" not in seen_titles:
        rebuilt_sections.extend(["## 间接传播链", *(indirect_lines or ["- （未结构化提供）"]), ""])

    return "\n".join(rebuilt_sections).strip()


def _build_impact_task_prompt(target: str, change_type: str) -> str:
    """构造 impact 任务提示词，强调先证据、后分层归类。"""
    return "\n".join(
        [
            f"请分析对 `{target}` 进行 `{change_type}` 变更的影响边界。",
            "",
            "任务要求:",
            "1. 这是一个“目标已知后的影响分析”任务，不是重新找目标的任务。",
            "2. 必须先使用 Serena 工具定位目标定义、直接引用点及关键边界符号。",
            "3. 必须把结果拆分为：直接调用点、数据契约/返回值消费者、间接传播链。",
            "4. 先写强证据，再补弱证据；如果弱证据不足，可少写，不要为了填满 section 去补推断。",
            "5. 最终必须调用 deliver(answer=..., confidence=...)；"
            "其中 answer 必须是结构化 Markdown 文本，不得输出 JSON、代码块或额外协议包装。",
            "6. 只返回已被工具证实的信息；无法确认的内容写入“不确定项”，不要编造。",
            "",
            "优先级方法:",
            "- 优先确认：直接调用、直接解构、直接传参、直接响应构造、直接请求解析。",
            "- 其次确认：前端接收点、前端继续传递点、后端下游关键函数。",
            "- 最后才考虑：UI 展示、派生统计、导出链路、按钮状态等弱传播链。",
            "- 如果某条关系只是“可能相关”但证据不够强，应放入“不确定项”或忽略，而不是写进主链。",
            "",
            "证据约束:",
            "- 只有被工具直接确认过的定位才能写成 `path:line`。",
            "- 如果只确认到文件，写 `path`，不要猜测行号。",
            "- 不要把目标定义文件的行号误写成调用点的行号。",
            "- 每条 bullet 只能绑定一个定位，说明必须只描述该定位所在文件中的直接行为。",
            "- 跨文件关系必须拆成多条，不要用上游文件的位置去承载下游文件的行为。",
            "- 在 deliver 前，应回读或检索你准备写出的每一个 `path:line`；无法复核的条目必须去掉行号。",
            "- 不要把按钮触发、页面入口、宏观流程误写成“直接调用点”，除非那里直接读取了目标符号或目标字段。",
            "- 对字段传播 / 数据契约类任务，同名的本地表单字段、派生状态、临时变量默认不是主传播链；必须先确认它直接来源于目标字段，才能写入主链路。",
            "- 若命中 `form.xxx`、局部 state、computed / derived value，仅能在确认其直接承接目标字段后再写入；否则应放入“不确定项”或忽略。",
            "",
            "deliver(answer=...) 的 Markdown 结构:",
            "## 直接调用点",
            "- `path:line` — 这里如何直接调用/解析/构造目标",
            "",
            "## 数据契约/返回值消费者",
            "- `path:line` — 哪个字段、返回值、解析点或消费者会直接受影响",
            "",
            "## 间接传播链",
            "- `path:line` — 下游传递、后续接口、UI 展示或其他间接消费",
            "",
            "## 边界符号",
            "- `symbol` — `kind` — `path:line`",
            "",
            "## 风险",
            "- 真实风险或回归面",
            "",
            "## 不确定项",
            "- 仅记录工具证据不足的事项",
            "",
            "## 建议修改顺序",
            "1. ...",
            "",
            "额外要求:",
            "- `直接调用点` 只保留真正直接受影响的位置，不要把间接上下游混入这一节。",
            "- `数据契约/返回值消费者` 优先覆盖：后端响应构造点、后端解析点、前端接收点、前端继续传递点。",
            "- `间接传播链` 再记录 UI 消费点、后续流程和更远的传播链。",
            "- `边界符号` 只保留关键定义点。",
            "- `风险` 区块只放自然语言总结，不要嵌套结构化对象。",
            "- 如果某个 section 证据不足，允许只写一两条高价值条目；不要把“可能有关”的弱链路硬塞进去。",
        ]
    )


def _build_impact_strategy_prompt(target: str, change_type: str) -> str:
    """构造 impact 前置策略画像请求。"""
    return "\n".join(
        [
            "请为下面这个 impact 任务生成一份很短的“策略画像”。",
            "",
            f"- target: {target}",
            f"- change_type: {change_type}",
            "",
            "输出格式固定为：",
            "## 策略画像",
            "- task_profile: symbol-first | dataflow-first | mixed",
            "- primary_evidence: 先取哪类证据",
            "- secondary_evidence: 再补哪类证据",
            "- tool_guidance: 推荐的工具使用倾向（强调优先级，不是禁令）",
            "- pitfalls: 最容易犯的偏差或误判",
            "",
            "## 首轮证据计划",
            "1. 先确认的第一类证据",
            "2. 先确认的第二类证据",
            "3. 先确认的第三类证据",
            "",
            "## 首轮检索键",
            "1. 第一个最值得先搜的字面量或标识符",
            "2. 第二个最值得先搜的字面量或标识符",
            "3. 第三个最值得先搜的字面量或标识符",
            "",
            "限制：",
            "- 每条只写一句",
            "- 不要写文件行号",
            "- 不要输出最终分析结论",
            "- 若任务涉及字段传播或数据契约，请显式提醒区分“远端响应字段/请求字段”与“同名本地表单字段/派生状态”。",
        ]
    )


def _derive_strategy_search_keys(target: str) -> list[str]:
    """从 target 中提取保守的默认检索键。"""
    keys: list[str] = []
    seen: set[str] = set()

    def add_key(value: str) -> None:
        token = value.strip().strip("`'\"")
        lowered = token.lower()
        if not token or lowered in seen or lowered in _GENERIC_STRATEGY_KEYS or len(token) < 3:
            return
        seen.add(lowered)
        keys.append(token)

    for endpoint in re.findall(r"/api/[A-Za-z0-9_./-]+", target):
        add_key(endpoint)
    for identifier in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", target):
        add_key(identifier)
    return keys[:3]


def _normalize_impact_strategy(text: str, *, target: str = "") -> str:
    """将前置策略画像标准化为可注入的短 Markdown。"""
    stripped = (text or "").strip()
    fallback_keys = _derive_strategy_search_keys(target)
    if not stripped:
        return "\n".join(
            [
                "## 策略画像",
                "- task_profile: mixed",
                "- primary_evidence: 先确认直接调用/定义点，再确认字段或返回值消费者",
                "- secondary_evidence: 补充下游传播链和 UI 消费点",
                "- tool_guidance: 先按更可靠的证据类型取证，再视结果补另一类工具",
                "- pitfalls: 不要过早总结，也不要把某一类工具当成唯一来源；同名本地状态不自动等于目标字段",
                "",
                "## 首轮证据计划",
                "1. 先确认目标定义点或字段构造点。",
                "2. 先确认最直接的调用点、解析点或接收点。",
                "3. 再确认下游传递点、关键消费者或 UI 展示点。",
                "",
                "## 首轮检索键",
                *[
                    f"{idx}. {key}"
                    for idx, key in enumerate(fallback_keys or ["layer_order"], start=1)
                ],
            ]
        )
    if "## 策略画像" not in stripped:
        lines = [line.strip("- ").strip() for line in stripped.splitlines() if line.strip()]
        stripped = "\n".join(["## 策略画像", *[f"- {line}" for line in lines[:5]]])
    if "## 首轮证据计划" not in stripped:
        stripped = "\n\n".join(
            [
                stripped,
                "\n".join(
                    [
                        "## 首轮证据计划",
                        "1. 先确认目标定义点或字段构造点。",
                        "2. 先确认最直接的调用点、解析点或接收点。",
                        "3. 再确认下游传递点、关键消费者或 UI 展示点。",
                    ]
                ),
            ]
        )
    if "## 首轮检索键" not in stripped:
        stripped = "\n\n".join(
            [
                stripped,
                "\n".join(
                    [
                        "## 首轮检索键",
                        *[
                            f"{idx}. {key}"
                            for idx, key in enumerate(fallback_keys or ["layer_order"], start=1)
                        ],
                    ]
                ),
            ]
        )
    return stripped


def _render_query_markdown(body: str, *, fallback_note: str | None = None) -> str:
    body = body.strip() or "未能生成有效回答。"
    lines = [
        "## 结论",
        body,
        "",
        "## 关键证据",
        "- （未结构化提供）",
        "",
        "## 相关符号",
        "- （未结构化提供）",
        "",
        "## 相关文件",
        "- （未结构化提供）",
        "",
        "## 不确定项",
        f"- {fallback_note or '输出未按协议分段，上述内容来自原始 deliver 文本。'}",
    ]
    return "\n".join(lines).strip()


def _render_impact_markdown(body: str, *, fallback_note: str | None = None) -> str:
    body = body.strip() or "分析结果无法解析。"
    lines = [
        "## 直接调用点",
        "- （未结构化提供）",
        "",
        "## 数据契约/返回值消费者",
        "- （未结构化提供）",
        "",
        "## 间接传播链",
        "- （未结构化提供）",
        "",
        "## 边界符号",
        "- （未结构化提供）",
        "",
        "## 风险",
        f"- {body}",
        "",
        "## 不确定项",
        f"- {fallback_note or '输出未按协议分段，上述内容来自原始 deliver 文本。'}",
        "",
        "## 建议修改顺序",
        "1. 重新运行 impact，并要求按 Markdown 协议输出",
    ]
    return "\n".join(lines).strip()


def _extract_strategy_search_keys(strategy_markdown: str) -> list[str]:
    """从策略块中提取首轮检索键。"""
    keys: list[str] = []
    in_section = False
    for raw_line in strategy_markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            in_section = line == "## 首轮检索键"
            continue
        if not in_section or not line:
            continue
        match = re.match(r"^(?:[-*]|\d+\.)\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).strip()
        if key and key not in keys:
            keys.append(key)
    return keys[:3]


def _extract_pattern_terms(pattern: str) -> list[str]:
    """从检索键中提取较稳定的语义 token，用于轻量打分与注释。"""
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        token = value.strip().strip("`'\"").lower()
        if not token or token in seen or len(token) < 3:
            return
        seen.add(token)
        terms.append(token)

    lowered = pattern.strip().strip("`'\"").lower()
    if lowered:
        add(lowered)
    for part in re.split(r"[^a-z0-9_/-]+", lowered):
        add(part)
        if "/" in part:
            for chunk in part.split("/"):
                add(chunk)
        if "-" in part:
            add(part.replace("-", "_"))
            for chunk in part.split("-"):
                add(chunk)
        if "_" in part:
            for chunk in part.split("_"):
                add(chunk)
    return terms[:8]


def _has_token_with_boundaries(text: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text) is not None


def _count_token_hits(text: str, token: str) -> int:
    return len(re.findall(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text))


def _looks_like_object_key(snippet: str, token: str) -> bool:
    return re.search(rf'["\']{re.escape(token)}["\']\s*:', snippet) is not None


def _looks_like_assignment_flow(snippet: str, token: str) -> bool:
    if "=" not in snippet:
        return False
    return _has_token_with_boundaries(snippet, token)


def _looks_like_call_argument_flow(snippet: str, token: str) -> bool:
    return re.search(rf"\([^)]*{re.escape(token)}[^)]*\)", snippet) is not None


def _looks_like_local_default(snippet: str, token: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])\s*[:=]\s*(?:['\"\[\{{]?\s*[\]}}'\"0-9a-z_-]*\s*)$",
            snippet,
        )
        is not None
    )


def _looks_like_member_access(snippet: str, token: str) -> bool:
    return re.search(rf"(?:\.|\[)\s*['\"]?{re.escape(token)}['\"]?\s*(?:\]|$)", snippet) is not None


def _classify_semantic_relation(pattern: str, snippet: str) -> tuple[int, str]:
    """基于结构关系而非技术栈 token 的 very light 语义判断。"""
    lowered = snippet.lower()
    terms = _extract_pattern_terms(pattern)
    if not terms:
        return 0, ""

    score = 0
    evidence_tags: list[str] = []

    for term in terms:
        if _has_token_with_boundaries(lowered, term):
            score += 2
        hits = _count_token_hits(lowered, term)
        if hits > 1:
            score += 1
        if _looks_like_object_key(lowered, term):
            score += 4
            evidence_tags.append("structured-key")
        if _looks_like_assignment_flow(lowered, term):
            score += 2
            evidence_tags.append("assignment-flow")
        if _looks_like_call_argument_flow(lowered, term):
            score += 2
            evidence_tags.append("call-argument")
        if _looks_like_member_access(lowered, term):
            score += 1
        if _looks_like_local_default(lowered, term):
            score -= 4
            evidence_tags.append("local-default")

    note = ""
    if "local-default" in evidence_tags and score <= 1:
        note = "（疑似局部默认值/弱关联点）"
    elif "structured-key" in evidence_tags:
        note = "（疑似结构化键位/契约点）"
    elif "call-argument" in evidence_tags:
        note = "（疑似参数传递点）"
    elif "assignment-flow" in evidence_tags:
        note = "（疑似值承接点）"
    return score, note


def _classify_pattern_match(pattern: str, snippet: str) -> tuple[int, str]:
    """对单条 pattern 命中做 very light 语义分类。"""
    return _classify_semantic_relation(pattern, snippet)


def _score_pattern_match(pattern: str, snippet: str) -> int:
    """为 search_for_pattern 的单条命中打分，优先保留更像主传播链的条目。"""
    score, _ = _classify_pattern_match(pattern, snippet)
    return score


def _pick_best_pattern_match(pattern: str, matches: list[Any]) -> str | None:
    """从同一文件的多条 pattern 命中中选出最有代表性的一条。"""
    best_snippet: str | None = None
    best_score = -(10**9)
    for raw in matches:
        snippet = str(raw).strip()
        score = _score_pattern_match(pattern, snippet)
        if best_snippet is None or score > best_score:
            best_snippet = snippet
            best_score = score
    return best_snippet


def _summarize_search_hits(pattern: str, raw: Any) -> str:
    """将 search_for_pattern 结果压缩为短 Markdown 片段。"""
    if not isinstance(raw, dict) or not raw:
        return f"- 搜索 `{pattern}`：未找到直接命中"

    lines = [f"- 搜索 `{pattern}`："]
    shown = 0
    for path, matches in raw.items():
        if shown >= 3:
            break
        if not isinstance(matches, list) or not matches:
            continue
        best = _pick_best_pattern_match(pattern, matches)
        if best is None:
            continue
        snippet = best.strip()
        snippet = snippet[:160] + "..." if len(snippet) > 160 else snippet
        _, note = _classify_pattern_match(pattern, snippet)
        lines.append(f"  - `{path}` — {snippet}{note}")
        shown += 1
    if shown == 0:
        return f"- 搜索 `{pattern}`：未找到直接命中"
    return "\n".join(lines)


# ============================================================================
# 响应解析
# ============================================================================


def _parse_query_response(content: str, *, project_root: str | Path | None = None) -> QueryResponse:
    """将 Agent 输出解析为 QueryResponse。"""
    _FALLBACK_ANSWERS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成推理，请缩小问题范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 推理超出总时长预算，请缩小问题范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，推理异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，推理可能不完整，请重试。",
        "__REACT_INPUT_TOO_LARGE__": "Agent 输入上下文超出预算且降级后仍过大，请缩小问题范围后重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小问题范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
        "__REACT_LLM_EXHAUSTED__": "Agent 的模型降级链已全部失败，请检查模型配置或限流状态后重试。",
    }
    if content in _FALLBACK_ANSWERS:
        return QueryResponse(markdown=_render_query_markdown(_FALLBACK_ANSWERS[content]))

    text = str(content).strip() if content else ""
    if _looks_like_markdown_sections(text):
        return QueryResponse(
            markdown=_post_validate_markdown_locations(text, project_root=project_root)
        )

    return QueryResponse(markdown=_render_query_markdown(text))


def _parse_impact_response(
    content: str, *, project_root: str | Path | None = None, target: str = ""
) -> ImpactResponse:
    """将 Agent 输出解析为 ImpactResponse。"""
    _FALLBACK_RISKS = {
        "__REACT_MAX_STEPS_EXCEEDED__": "Agent 未能在限定步数内完成影响分析，请缩小分析范围后重试。",
        "__REACT_TIMEOUT_BUDGET__": "Agent 分析超出总时长预算，请缩小分析范围后重试。",
        "__REACT_NO_TOOL_EXHAUSTED__": "Agent 多次未调用工具，分析异常终止，请重试。",
        "__REACT_DELIVER_EMPTY__": "Agent 提交了空结论，分析可能不完整，请重试。",
        "__REACT_INPUT_TOO_LARGE__": "Agent 输入上下文超出预算且降级后仍过大，请缩小分析范围后重试。",
        "__REACT_LENGTH_EXHAUSTED__": "Agent 输出被多次截断且续写次数耗尽，请重试或缩小分析范围。",
        "__REACT_TIMEOUT__": "Agent 调用模型超时且重试耗尽，请稍后重试。",
        "__REACT_LLM_EXHAUSTED__": "Agent 的模型降级链已全部失败，请检查模型配置或限流状态后重试。",
    }
    if content in _FALLBACK_RISKS:
        return ImpactResponse(markdown=_render_impact_markdown(_FALLBACK_RISKS[content]))

    text = str(content).strip() if content else ""
    if _looks_like_markdown_sections(text):
        normalized = _post_validate_markdown_locations(text, project_root=project_root)
        return ImpactResponse(markdown=_post_filter_impact_markdown(normalized, target=target))

    return ImpactResponse(markdown=_render_impact_markdown(text))


# ============================================================================
# 主 Agent 类
# ============================================================================


class PCEAgent(BaseReActAgent):
    """PCE ReAct Agent。

    驱动 Serena 工具调用，返回推理结论。每次调用均从 Memory 快照重新起步（无状态）。
    """

    # PCEAgent 启用预算告警
    _enable_budget_warning: bool = True

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        model_fallbacks: list[str] | None = None,
        max_seconds: float = MAX_SECONDS,
        insight_cache: InsightCache | None = None,
    ) -> None:
        super().__init__(
            model=model,
            provider=provider,
            model_fallbacks=model_fallbacks,
            max_seconds=max_seconds,
        )
        self._deliver_tool = DELIVER_TOOL
        self._insight_cache = insight_cache

    # =========================================================================
    # BaseReActAgent 抽象方法实现
    # =========================================================================

    def build_tools_schema(
        self,
        serena_client: SerenaClient,
        state: LoopState,
    ) -> list[dict[str, Any]]:
        tools = [*serena_client.tools_schema, self._deliver_tool]
        if state.extra.get("depth", 0) == 0:
            tools.append(SPAWN_AGENT_TOOL)
        if state.extra.get("acknowledge_cb") is not None:
            tools.append(ACKNOWLEDGE_TOOL)
        return tools

    @property
    def virtual_tool_names(self) -> set[str]:
        return {"spawn_agent", "acknowledge_changes"}

    async def handle_virtual_tool(
        self,
        tool_call: Any,
        state: LoopState,
    ) -> dict[str, Any] | None:
        name = _get_tool_name(tool_call)
        tc_id = _get_tool_call_id(tool_call)
        args = _extract_tool_call_args(tool_call)
        req_id = _req_id_var.get()

        # ── acknowledge_changes ──
        if name == "acknowledge_changes":
            paths = (args or {}).get("paths", []) or []
            acknowledge_cb = state.extra.get("acknowledge_cb")
            if acknowledge_cb and paths:
                try:
                    await acknowledge_cb(paths)
                    content = f"已确认 {len(paths)} 个文件的认知状态"
                except Exception as exc:
                    content = f"认知确认失败: {exc}"
            else:
                content = "无需确认（无变更文件或回调未配置）"
            logger.info(
                "[req=%s] round=%d -> acknowledge_changes paths=%d",
                req_id, state.round_num, len(paths),
            )
            return {"tool_call_id": tc_id, "name": "acknowledge_changes", "content": content}

        # ── spawn_agent ──
        if name != "spawn_agent":
            return None

        serena_client: SerenaClient = state.extra["serena_client"]
        depth = int(state.extra.get("depth", 0))
        spawn_count = int(state.extra.get("spawn_count", 0))

        if args is None:
            spawn_result = SpawnResult(
                ok=False, task_id=str(uuid.uuid4()),
                status=SpawnStatus.FAILED, answer="", confidence="low",
                elapsed_seconds=0.0,
                error_code=SpawnErrorCode.SUBAGENT_INVALID_ARGS,
                error_message="spawn_agent 参数解析失败：无效 JSON",
            )
        elif spawn_count >= MAX_SPAWNS_PER_LOOP:
            spawn_result = SpawnResult(
                ok=False, task_id=str(uuid.uuid4()),
                status=SpawnStatus.FAILED, answer="", confidence="low",
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
                    context=args["context"] if isinstance(args.get("context"), dict) else {},
                    strict=bool(args.get("strict", False)),
                )
                if not request.task:
                    raise ValueError("task 不能为空")
            except Exception as exc:
                spawn_result = SpawnResult(
                    ok=False, task_id=str(uuid.uuid4()),
                    status=SpawnStatus.FAILED, answer="", confidence="low",
                    elapsed_seconds=0.0,
                    error_code=SpawnErrorCode.SUBAGENT_INVALID_ARGS,
                    error_message=f"spawn_agent 参数无效: {exc}",
                )
            else:
                spawn_result = await invoke_spawn(
                    request=request,
                    serena_client=serena_client,
                    parent_depth=depth,
                    parent_deadline=state.deadline,
                    run_loop_fn=self.run_loop,
                )
                state.extra["spawn_count"] = spawn_count + 1
                if spawn_result.ok:
                    await self._persist_insights(
                        question=request.task,
                        answer=spawn_result.answer,
                        confidence=spawn_result.confidence,
                    )

        logger.info(
            "[req=%s] round=%d -> spawn_agent ok=%s %.1fs",
            req_id, state.round_num, spawn_result.ok, spawn_result.elapsed_seconds,
        )
        return {
            "tool_call_id": tc_id,
            "name": "spawn_agent",
            "content": _safe_json_dumps(spawn_result.to_tool_content()),
        }

    async def on_deliver(
        self,
        args: dict[str, Any] | None,
        state: LoopState,
    ) -> DeliverDecision:
        if args is None:
            logger.warning("deliver 调用参数解析失败")
            return DeliverDecision.finish(REACT_DELIVER_EMPTY)
        answer = args.get("answer")
        if not answer:
            logger.warning("deliver 调用缺少 answer 参数")
            return DeliverDecision.finish(REACT_DELIVER_EMPTY)

        confidence = args.get("confidence")
        req_id = _req_id_var.get()
        logger.info(
            "[req=%s] DELIVER confidence=%s elapsed=%.1fs rounds=%d",
            req_id, confidence, state.elapsed, state.round_num,
        )
        trace = state.extra.get("trace")
        if trace:
            await trace.write({
                "event": "deliver",
                "confidence": confidence,
                "elapsed_seconds": state.elapsed,
                "rounds": state.round_num,
                "answer": str(answer),
            })
        return DeliverDecision.finish(str(answer), str(confidence) if confidence else None)

    # =========================================================================
    # BaseReActAgent 钩子覆盖
    # =========================================================================

    async def on_before_round(self, state: LoopState) -> None:
        """每轮 LLM 调用前执行 compact 检查。"""
        new_messages = await self._maybe_compact(state)
        if new_messages is not None:
            self._replace_messages(state, new_messages)
            trace = state.extra.get("trace")
            if trace:
                for msg in new_messages:
                    await trace.write({"event": "compact_rebuild", **msg})

    async def _invoke_serena(
        self,
        tool_call: Any,
        serena_client: SerenaClient,
        state: LoopState | None = None,
    ) -> dict[str, Any]:
        """增强版 Serena 调用：添加计时与 preview 日志。"""
        tc_id = _get_tool_call_id(tool_call)
        tool_name = _get_tool_name(tool_call) or "unknown"
        args = _extract_tool_call_args(tool_call)
        preview = _key_arg_preview(tool_name, args)
        req_id = _req_id_var.get()
        round_num = state.round_num if state is not None else 0

        if args is None:
            logger.warning("工具参数解析失败: %s", tool_name)
            content = f"工具参数解析失败: 无效的 JSON。请重新调用 {tool_name} 并确保参数格式正确。"
            logger.info(
                "[req=%s] round=%d -> %s 0.00s %dchars",
                req_id, round_num, _key_arg_preview(tool_name, None), len(content),
            )
            return {"tool_call_id": tc_id, "name": tool_name, "content": content}

        blocked_reason = self._blocked_serena_access_reason(
            tool_name,
            args,
            project_root=serena_client.project_path,
        )
        if blocked_reason is not None:
            logger.info(
                "[req=%s] round=%d -> %s blocked %dchars",
                req_id, round_num, preview, len(blocked_reason),
            )
            return {"tool_call_id": tc_id, "name": tool_name, "content": blocked_reason}

        t0 = time.monotonic()
        try:
            result = await serena_client.call(tool_name, args)
            content = _safe_json_dumps(result)
        except SerenaClientError as exc:
            logger.warning("工具调用失败: %s: %s", tool_name, exc)
            content = f"工具调用失败: {exc}"

        elapsed = time.monotonic() - t0
        logger.info(
            "[req=%s] round=%d -> %s %.2fs %dchars",
            req_id, round_num, preview, elapsed, len(content),
        )
        return {"tool_call_id": tc_id, "name": tool_name, "content": content}

    @staticmethod
    def _blocked_serena_access_reason(
        tool_name: str,
        args: dict[str, Any],
        *,
        project_root: Path,
    ) -> str | None:
        del tool_name
        relative_path = args.get("relative_path")
        if not isinstance(relative_path, str):
            return None
        normalized = relative_path.strip()
        if not normalized:
            return None
        if is_visible_to_agent(project_root, normalized):
            return None
        return (
            f"路径 `{normalized}` 不在当前 PCE Agent 的可读边界内。"
            "该路径命中了项目 ignore / PCE ignore / 安全围栏，因此不可通过 Serena 读取或搜索。"
        )

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
            async with aiofiles.open(path, encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return f"(未找到 {path.name},索引尚未构建)"
        except Exception as e:
            logger.warning(f"读取文件失败: {path}: {e}")
            return f"(读取 {path.name} 失败)"

    @staticmethod
    def _compact_markdown_block(content: str) -> str:
        """对静态注入块做格式级紧凑化，不改变语义内容。"""
        normalized: list[str] = []
        previous_blank = False
        for raw in content.splitlines():
            line = raw.rstrip()
            blank = not line.strip()
            if blank:
                if previous_blank or not normalized:
                    continue
                previous_blank = True
                normalized.append("")
                continue
            previous_blank = False
            normalized.append(line)
        while normalized and not normalized[-1].strip():
            normalized.pop()
        return "\n".join(normalized).strip()

    @staticmethod
    def _compose_system_prompt(structure_md: str, annotations_index_md: str, injected: str) -> str:
        """将各注入块拼装为完整 system prompt 文本。"""
        sections = [
            SYSTEM_PROMPT_HEADER,
            "",
            "## 项目结构 (structure.md)",
            structure_md.strip(),
            "",
            "## 项目认知导航 (annotations/index.md)",
            annotations_index_md.strip(),
            "",
            "## 认知导航路由说明",
            "- 若 index.md 只给出区域入口，先读取对应的 `annotations/areas/{area}.md`。",
            "- 在区域文档中根据模块列表继续下钻到 `annotations/modules/{module}.md`。",
            "- 若当前仓库仍是旧布局、找不到 `areas/*.md`，再直接进入 `annotations/modules/*.md`。",
            "- 回答具体实现问题前，不要停留在 area 层；需要继续检索模块文档或调用工具取证。",
        ]
        if injected.strip():
            sections += [
                "",
                "## 动态认知提示",
                "以下内容来自历史会话蒸馏，可能过时；必须通过工具验证后再使用。",
                "",
                injected.strip(),
            ]
        return "\n".join(sections)

    async def _build_system_prompt(self, memory_root: Path | None) -> str:
        """构建 system prompt,注入 structure.md 和 annotations/index.md 内容。

        annotations/index.md 是轻量的项目/区域导航入口；区域文档（areas/*.md）与
        各模块深度认知文档（modules/*.md）由 PCE Agent 在 ReAct 循环中按需加载。

        若 insight_cache 已配置，在末尾追加动态认知块（top-k 条目）。

        超出 _SYSTEM_PROMPT_SOFT_LIMIT 时按以下顺序降级：
          降级1 — 截断 structure.md（保留比例行数 + 截断提示）
          降级2 — 截断 annotations/index.md
          降级3 — 两者均替换为占位文本
        每级降级均打 WARNING 日志标明触发块与 token 估算值。
        """
        root = memory_root or Path.cwd()
        pce_dir = root / ".pce"

        structure_md = self._compact_markdown_block(
            await self._read_pce_file(pce_dir / "structure.md")
        )
        annotations_index_md = self._compact_markdown_block(
            await self._read_pce_file(pce_dir / "annotations" / "index.md")
        )

        injected = ""
        if self._insight_cache is not None:
            try:
                injected, _ = await self._insight_cache.get_top_k(
                    k=_INSIGHT_TOP_K,
                    token_budget=_INSIGHT_TOKEN_BUDGET,
                )
            except Exception as e:
                logger.warning(f"读取 Insight Cache 失败，忽略本轮注入: {e}")

        prompt = self._compose_system_prompt(structure_md, annotations_index_md, injected)

        # 正常路径：只对可压缩的动态注入块做 token 估算（不含 HEADER 和工具 schema）
        # HEADER 和工具 schema 不可压缩、不应参与上限判断
        dynamic_content = "\n\n".join(
            filter(
                None,
                [
                    structure_md.strip(),
                    annotations_index_md.strip(),
                    injected.strip(),
                ],
            )
        )
        token_count = litellm.token_counter(
            model=self._model,
            messages=[{"role": "system", "content": dynamic_content}],
        )
        if token_count <= _SYSTEM_PROMPT_SOFT_LIMIT:
            return prompt

        def _truncate_by_ratio(content: str, budget_ratio: float) -> str:
            """按比例保留前 N 行；实际截断时末尾追加截断提示。"""
            lines = content.strip().splitlines()
            if not lines or budget_ratio >= 1.0:
                return content.strip()
            keep = max(1, int(len(lines) * budget_ratio))
            if keep >= len(lines):
                return content.strip()
            return "\n".join(lines[:keep]) + _SYSTEM_PROMPT_TRUNCATED_NOTICE

        # 目标：将每个动态块各自压缩到 ratio 比例，使总 token 回到软上限以内
        # ratio 直接作用于每个块自身，不做跨块分配（降级1只压 structure，降级2再压 annotations）
        ratio = (_SYSTEM_PROMPT_SOFT_LIMIT / token_count) * 0.85  # 留 15% 余量

        def _count_dynamic(s: str, a: str, i: str) -> int:
            """只统计动态注入块的 token 数（不含 HEADER 和工具 schema）。"""
            content = "\n\n".join(filter(None, [s.strip(), a.strip(), i.strip()]))
            return litellm.token_counter(
                model=self._model,
                messages=[{"role": "system", "content": content}],
            )

        # 降级1：截断 structure.md
        structure_truncated = _truncate_by_ratio(structure_md, ratio)
        token_count = _count_dynamic(structure_truncated, annotations_index_md, injected)
        logger.warning(
            "system prompt 超出软上限，触发降级1: block=structure.md, tokens=%d",
            token_count,
        )
        if token_count <= _SYSTEM_PROMPT_SOFT_LIMIT:
            return self._compose_system_prompt(structure_truncated, annotations_index_md, injected)

        # 降级2：同时截断 annotations/index.md
        annotations_truncated = _truncate_by_ratio(annotations_index_md, ratio)
        token_count = _count_dynamic(structure_truncated, annotations_truncated, injected)
        logger.warning(
            "system prompt 超出软上限，触发降级2: block=annotations/index.md, tokens=%d",
            token_count,
        )
        if token_count <= _SYSTEM_PROMPT_SOFT_LIMIT:
            return self._compose_system_prompt(structure_truncated, annotations_truncated, injected)

        # 降级3：两者均替换为占位文本
        logger.warning(
            "system prompt 超出软上限，触发降级3: 全部动态块替换为占位文本, tokens=%d",
            token_count,
        )
        return self._compose_system_prompt(
            _SYSTEM_PROMPT_PLACEHOLDER,
            _SYSTEM_PROMPT_PLACEHOLDER,
            injected,
        )

    def _placeholder_system_prompt(self) -> str:
        return self._compose_system_prompt(
            _SYSTEM_PROMPT_PLACEHOLDER,
            _SYSTEM_PROMPT_PLACEHOLDER,
            "",
        )

    def _build_initial_tools_schema(
        self,
        serena_client: SerenaClient,
        acknowledge_cb: AcknowledgeCallback | None,
    ) -> list[dict[str, Any]]:
        now = time.monotonic()
        state = LoopState(messages=[], start_time=now, deadline=now + 1.0)
        state.extra["depth"] = 0
        state.extra["acknowledge_cb"] = acknowledge_cb
        return self.build_tools_schema(serena_client, state)

    def _guard_query_prompt(
        self,
        *,
        system_content: str,
        question: str,
        tools_schema: list[dict[str, Any]],
    ) -> tuple[str, str]:
        budget = build_prompt_budget(_CONTEXT_WINDOW)
        user_content = question.strip()
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        tokens = estimate_input_tokens(self._model, messages, tools_schema)
        if tokens <= budget.soft_input_budget:
            return system_content, user_content

        user_budget = max(600, budget.target_input_budget // 4)
        user_content = fit_text_to_budget(
            self._model,
            user_content,
            token_budget=user_budget,
            notice="\n\n[用户输入过长，已保留前后关键部分]\n\n",
        )
        messages[1]["content"] = user_content
        tokens = estimate_input_tokens(self._model, messages, tools_schema)
        if tokens <= budget.hard_input_budget:
            logger.warning("query 初始输入触发降级: action=truncate_user tokens=%d", tokens)
            return system_content, user_content

        system_content = self._placeholder_system_prompt()
        messages[0]["content"] = system_content
        remaining_budget = max(400, budget.target_input_budget // 5)
        user_content = fit_text_to_budget(
            self._model,
            user_content,
            token_budget=remaining_budget,
            notice="\n\n[上下文预算不足，已进一步压缩用户输入]\n\n",
        )
        logger.warning("query 初始输入触发降级: action=placeholder_system")
        return system_content, user_content

    def _guard_impact_prompt(
        self,
        *,
        system_content: str,
        target: str,
        change_type: str,
        strategy_note: str,
        evidence_seed_note: str,
        tools_schema: list[dict[str, Any]],
    ) -> tuple[str, str]:
        budget = build_prompt_budget(_CONTEXT_WINDOW)
        task_prompt = _build_impact_task_prompt(target=target, change_type=change_type)
        execution_note = "\n".join(
            [
                "执行要求补充：",
                "- 在开始总结前，优先完成“首轮证据计划”中的确认动作。",
                "- 若首轮证据不足，再自行补充工具取证，不要机械拘泥于策略块。",
            ]
        )

        def _render(strategy_text: str, evidence_text: str) -> str:
            blocks = [
                task_prompt,
                "以下是本轮任务的前置策略画像，请将其视为取证顺序建议，而不是硬性禁令：",
                strategy_text.strip(),
            ]
            if evidence_text.strip():
                blocks.extend(
                    [
                        "以下是根据首轮检索键预采集的证据种子，请优先消化这些证据后再扩展检索：",
                        evidence_text.strip(),
                    ]
                )
            blocks.append(execution_note)
            return "\n\n".join(block for block in blocks if block.strip())

        user_content = _render(strategy_note, evidence_seed_note)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        tokens = estimate_input_tokens(self._model, messages, tools_schema)
        if tokens <= budget.soft_input_budget:
            return system_content, user_content

        user_content = _render(strategy_note, "")
        messages[1]["content"] = user_content
        tokens = estimate_input_tokens(self._model, messages, tools_schema)
        if tokens <= budget.hard_input_budget:
            logger.warning("impact 初始输入触发降级: action=drop_evidence_seed tokens=%d", tokens)
            return system_content, user_content

        minimal_strategy = _normalize_impact_strategy("", target=target)
        user_content = _render(minimal_strategy, "")
        messages[1]["content"] = user_content
        tokens = estimate_input_tokens(self._model, messages, tools_schema)
        if tokens <= budget.hard_input_budget:
            logger.warning("impact 初始输入触发降级: action=minimal_strategy tokens=%d", tokens)
            return system_content, user_content

        system_content = self._placeholder_system_prompt()
        messages[0]["content"] = system_content
        user_content = fit_text_to_budget(
            self._model,
            user_content,
            token_budget=max(700, budget.target_input_budget // 3),
            notice="\n\n[影响分析提示过长，已保留关键任务描述]\n\n",
        )
        logger.warning("impact 初始输入触发降级: action=placeholder_system")
        return system_content, user_content

    async def _maybe_compact(self, state: LoopState) -> list[dict[str, Any]] | None:
        """在上下文使用率达到阈值时触发 compact，蒸馏认知后重建对话窗口。

        通过 state.token_used 和 state.extra["compact_failed"] 管理状态。
        成功时返回重建后的消息列表并更新 state；未触发或失败时返回 None。
        """
        token_used = state.token_used
        compact_failed = bool(state.extra.get("compact_failed", False))

        if (
            _CONTEXT_WINDOW <= 0
            or compact_failed
            or token_used / _CONTEXT_WINDOW < _COMPACT_THRESHOLD
        ):
            return None

        logger.info(
            f"上下文使用率 {token_used}/{_CONTEXT_WINDOW} "
            f"({token_used / _CONTEXT_WINDOW:.1%}) 达到阈值，触发 compact"
        )

        system_msgs = [m for m in state.messages if m.get("role") == "system"]
        first_user = next((m for m in state.messages if m.get("role") == "user"), None)
        task_hint = f"\n\n当前任务：{first_user['content']}" if first_user else ""

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
        budget = build_prompt_budget(_CONTEXT_WINDOW)
        compact_tokens = estimate_input_tokens(self._model, compact_request, [])
        if compact_tokens > budget.hard_input_budget:
            if len(system_msgs) > 1:
                compact_request = [system_msgs[-1], compact_request[-1]]
            compact_request[-1]["content"] = fit_text_to_budget(
                self._model,
                str(compact_request[-1]["content"]),
                token_budget=max(800, budget.target_input_budget // 3),
                notice="\n\n[compact 请求已压缩]\n\n",
            )
            logger.warning(
                "compact 请求超出预算，触发预压缩: tokens=%d hard=%d",
                compact_tokens,
                budget.hard_input_budget,
            )

        try:
            summary_response, _ = await self._completion(compact_request, [])
            summary_msg = _extract_message(summary_response)
            summary_text = str(summary_msg.get("content") or "").strip()
            compact_tokens = self._extract_next_prompt_size(summary_response)
        except LLMCompletionError:
            raise
        except Exception as exc:
            logger.warning(f"compact 摘要生成失败，跳过压缩（本轮不再重试）: {exc}")
            state.extra["compact_failed"] = True
            return None

        if not summary_text:
            logger.warning("compact 摘要为空，跳过压缩（本轮不再重试）")
            state.extra["compact_failed"] = True
            return None

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
        state.token_used = compact_tokens
        state.extra["compact_failed"] = False
        return new_messages

    async def _plan_impact_strategy(
        self,
        *,
        target: str,
        change_type: str,
        trace: _TraceWriter | None = None,
    ) -> str:
        """生成 impact 前置策略画像。失败时返回保守的默认策略。"""
        prompt = _build_impact_strategy_prompt(target=target, change_type=change_type)
        messages = [
            {"role": "system", "content": IMPACT_STRATEGY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response, _ = await self._completion(messages, [])
            message = _extract_message(response)
            strategy = _normalize_impact_strategy(str(message.get("content") or ""), target=target)
            if trace:
                await trace.write({"role": "system", "content": IMPACT_STRATEGY_SYSTEM_PROMPT})
                await trace.write({"role": "user", "content": prompt})
                await trace.write({"event": "impact_strategy", "content": strategy})
            return strategy
        except Exception as exc:
            logger.warning("impact 前置策略画像失败，降级为默认策略: %s", exc)
            strategy = _normalize_impact_strategy("", target=target)
            if trace:
                await trace.write(
                    {
                        "event": "impact_strategy_fallback",
                        "reason": str(exc),
                        "content": strategy,
                    }
                )
            return strategy

    async def _collect_impact_evidence_seeds(
        self,
        *,
        strategy_markdown: str,
        serena_client: SerenaClient,
        trace: _TraceWriter | None = None,
    ) -> str:
        """依据策略块中的首轮检索键做轻量检索，生成证据种子。"""
        search_keys = _extract_strategy_search_keys(strategy_markdown)
        if not search_keys:
            return "## 首轮证据种子\n- 未生成可执行的检索键，请主 Agent 自主规划取证。"

        bullets: list[str] = ["## 首轮证据种子"]
        for key in search_keys[:3]:
            try:
                raw = await serena_client.search_for_pattern(
                    key,
                    context_lines_before=1,
                    context_lines_after=1,
                )
                bullets.append(_summarize_search_hits(key, raw))
            except Exception as exc:
                bullets.append(f"- 搜索 `{key}`：执行失败（{exc}）")

        summary = "\n".join(bullets)
        if trace:
            await trace.write({"event": "impact_evidence_seeds", "content": summary})
        return summary

    # ============================================================================
    # 公开接口
    # ============================================================================

    async def query(
        self,
        question: str,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> QueryResponse:
        """执行自然语言查询。

        Args:
            question: 用户问题
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            Markdown 格式的查询响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        req_id = uuid.uuid4().hex[:8]
        token = _req_id_var.set(req_id)
        trace = _TraceWriter.from_env(req_id)
        try:
            logger.info('[req=%s] QUERY "%s"', req_id, question[:80])
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            tools_schema = self._build_initial_tools_schema(serena_client, acknowledge_cb)
            system_content, user_content = self._guard_query_prompt(
                system_content=system_content,
                question=question,
                tools_schema=tools_schema,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            if trace:
                await trace.write({"role": "system", "content": system_content})
                await trace.write({"role": "user", "content": user_content})

            answer, confidence = await self.run_loop(
                messages,
                serena_client,
                acknowledge_cb=acknowledge_cb,
                trace_writer=trace,
            )
            await self._persist_insights(question=question, answer=answer, confidence=confidence)
            return _parse_query_response(answer, project_root=serena_client.project_path)
        finally:
            if trace:
                await trace.rotate()
            _req_id_var.reset(token)

    async def impact(
        self,
        target: str,
        change_type: str,
        memory_root: str | Path | None = None,
        serena_client: SerenaClient | None = None,
        acknowledge_cb: AcknowledgeCallback | None = None,
    ) -> ImpactResponse:
        """执行变更影响分析。

        Args:
            target: 分析目标(符号名或文件路径)
            change_type: 变更类型(modify/rename/delete/add_field/change_signature)
            memory_root: Memory 文件根路径
            serena_client: 已连接的 SerenaClient
            acknowledge_cb: 认知确认回调,Agent 探索变更文件后触发

        Returns:
            Markdown 格式的影响分析响应

        Raises:
            SerenaClientError: serena_client 未提供
        """
        if serena_client is None:
            raise SerenaClientError("serena_client 未提供")

        req_id = uuid.uuid4().hex[:8]
        token = _req_id_var.set(req_id)
        trace = _TraceWriter.from_env(req_id)
        try:
            logger.info(
                '[req=%s] IMPACT target="%s" change_type=%s', req_id, target[:80], change_type
            )
            memory_path = Path(memory_root) if memory_root else None
            system_content = await self._build_system_prompt(memory_path)
            strategy_note = await self._plan_impact_strategy(
                target=target,
                change_type=change_type,
                trace=trace,
            )
            evidence_seed_note = await self._collect_impact_evidence_seeds(
                strategy_markdown=strategy_note,
                serena_client=serena_client,
                trace=trace,
            )
            tools_schema = self._build_initial_tools_schema(serena_client, acknowledge_cb)
            system_content, prompt = self._guard_impact_prompt(
                system_content=system_content,
                target=target,
                change_type=change_type,
                strategy_note=strategy_note,
                evidence_seed_note=evidence_seed_note,
                tools_schema=tools_schema,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

            if trace:
                await trace.write({"role": "system", "content": system_content})
                await trace.write({"role": "user", "content": prompt})

            answer, confidence = await self.run_loop(
                messages,
                serena_client,
                acknowledge_cb=acknowledge_cb,
                trace_writer=trace,
            )
            await self._persist_insights(question=prompt, answer=answer, confidence=confidence)
            return _parse_impact_response(
                answer,
                project_root=serena_client.project_path,
                target=target,
            )
        finally:
            if trace:
                await trace.rotate()
            _req_id_var.reset(token)
