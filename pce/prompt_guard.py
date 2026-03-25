"""Prompt 预算估算与降级辅助。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import litellm


@dataclass(frozen=True, slots=True)
class PromptBudget:
    context_window: int
    completion_reserve: int
    tool_reserve: int
    hard_input_budget: int
    soft_input_budget: int
    target_input_budget: int


def get_context_window() -> int:
    raw = os.getenv("PCE_CONTEXT_WINDOW", "256000")
    try:
        value = int(raw)
    except ValueError:
        return 256000
    return value if value > 0 else 256000


def build_prompt_budget(context_window: int | None = None) -> PromptBudget:
    window = context_window or get_context_window()
    completion_reserve = max(4096, int(window * 0.15))
    tool_reserve = max(2048, int(window * 0.05))
    hard = max(4000, window - completion_reserve - tool_reserve)
    soft = max(3000, int(hard * 0.9))
    target = max(2500, int(hard * 0.8))
    return PromptBudget(
        context_window=window,
        completion_reserve=completion_reserve,
        tool_reserve=tool_reserve,
        hard_input_budget=hard,
        soft_input_budget=soft,
        target_input_budget=target,
    )


def estimate_text_tokens(model: str, text: str) -> int:
    if not text.strip():
        return 0
    try:
        return litellm.token_counter(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
    except Exception:
        return max(1, len(text) // 4)


def estimate_input_tokens(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    try:
        message_tokens = litellm.token_counter(model=model, messages=messages)
    except Exception:
        message_tokens = max(1, len(json.dumps(messages, ensure_ascii=False)) // 4)

    if not tools:
        return message_tokens

    try:
        tools_text = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        tools_text = str(tools)
    tool_tokens = estimate_text_tokens(model, tools_text)
    return message_tokens + tool_tokens


def truncate_head_tail(
    text: str,
    *,
    max_chars: int,
    notice: str,
) -> str:
    raw = text.strip()
    if len(raw) <= max_chars:
        return raw
    if max_chars <= len(notice) + 20:
        return (raw[: max(0, max_chars - len(notice))] + notice).strip()
    head = max_chars // 2
    tail = max_chars - head - len(notice)
    return f"{raw[:head].rstrip()}{notice}{raw[-tail:].lstrip()}".strip()


def truncate_lines_by_ratio(
    text: str,
    *,
    ratio: float,
    notice: str,
) -> str:
    lines = text.strip().splitlines()
    if not lines or ratio >= 1.0:
        return text.strip()
    keep = max(1, int(len(lines) * ratio))
    if keep >= len(lines):
        return text.strip()
    return "\n".join(lines[:keep]).rstrip() + notice


def fit_text_to_budget(
    model: str,
    text: str,
    *,
    token_budget: int,
    notice: str,
    min_chars: int = 400,
) -> str:
    raw = text.strip()
    if not raw:
        return raw
    if estimate_text_tokens(model, raw) <= token_budget:
        return raw

    low = max(64, min_chars)
    high = len(raw)
    best = truncate_head_tail(raw, max_chars=low, notice=notice)
    if estimate_text_tokens(model, best) > token_budget:
        return best

    while low <= high:
        mid = (low + high) // 2
        candidate = truncate_head_tail(raw, max_chars=mid, notice=notice)
        if estimate_text_tokens(model, candidate) <= token_budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best
