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


def normalize_litellm_model(model: str | None, *, api_base: str | None = None) -> str | None:
    """按当前配置补全 LiteLLM provider 前缀。

    规则：
    - 若模型已显式包含 provider 前缀，则原样返回
    - 若设置了 PCE_LITELLM_PROVIDER，则补成 <provider>/<model>
    - 否则，只要配置了 api_base，就按最通用的 OpenAI 兼容端点补成 openai/<model>
    """
    if model is None:
        return None

    text = model.strip()
    if not text:
        return None

    provider_override = get_env_text("PCE_LITELLM_PROVIDER")
    if provider_override:
        provider = provider_override.strip()
        if provider and not text.startswith(f"{provider}/"):
            return f"{provider}/{text}"
        return text

    provider_prefixes = {
        "openai",
        "openrouter",
        "anthropic",
        "azure",
        "bedrock",
        "gemini",
        "vertex_ai",
        "huggingface",
        "ollama",
        "groq",
        "mistral",
        "deepseek",
        "fireworks_ai",
        "together_ai",
        "xai",
        "cerebras",
        "cohere",
        "replicate",
    }
    head = text.split("/", 1)[0]
    if head in provider_prefixes:
        return text

    effective_api_base = api_base or get_env_text("PCE_API_BASE")
    if effective_api_base:
        return f"openai/{text}"

    return text
