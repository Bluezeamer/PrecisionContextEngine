"""环境变量读取辅助。

集中处理 PCE 运行时对 litellm 的可选覆盖参数，供 agent.py 和 indexer.py 共享。
"""

from __future__ import annotations

import os
from typing import Any


def get_env_text(name: str) -> str | None:
    """读取环境变量文本，去除首尾空白；空字符串视为未配置，返回 None。"""
    value = os.getenv(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def get_completion_overrides() -> dict[str, Any]:
    """读取 litellm completion 的可选覆盖参数（每次调用时读取，以响应 _bootstrap 后加载的 .env）。

    返回值仅包含已配置的键：
    - api_key:  PCE_API_KEY，覆盖 litellm 的供应商默认 key
    - api_base: PCE_API_BASE，指定自定义 base URL（OpenAI 兼容端点）
    """
    overrides: dict[str, Any] = {}
    if (api_key := get_env_text("PCE_API_KEY")) is not None:
        overrides["api_key"] = api_key
    if (api_base := get_env_text("PCE_API_BASE")) is not None:
        overrides["api_base"] = api_base
    return overrides
