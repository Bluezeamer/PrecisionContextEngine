"""环境变量读取辅助。

集中处理 PCE 运行时对 LiteLLM 的最小配置：
- provider
- model
- base_url（可选）
- api_key（可选）
- temperature（可选）
"""

from __future__ import annotations

import os
from typing import Any


def get_env_text(name: str) -> str | None:
    """读取环境变量文本，去除首尾空白；空字符串视为未配置。"""
    value = os.getenv(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def get_base_url() -> str | None:
    """读取自定义 LLM Base URL。

    优先使用更直观的 `PCE_BASE_URL`；`PCE_API_BASE` 作为兼容别名保留。
    """
    return get_env_text("PCE_BASE_URL") or get_env_text("PCE_API_BASE")


def get_completion_overrides() -> dict[str, Any]:
    """读取 litellm completion 的可选覆盖参数。"""
    overrides: dict[str, Any] = {}
    if (api_key := get_env_text("PCE_API_KEY")) is not None:
        overrides["api_key"] = api_key
    if (api_base := get_base_url()) is not None:
        overrides["api_base"] = api_base
    return overrides


def get_env_float(name: str) -> float | None:
    """读取环境变量浮点数；非法值视为未配置。"""
    text = get_env_text(name)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def get_env_int(name: str) -> int | None:
    """读取环境变量整数；非法值视为未配置。"""
    text = get_env_text(name)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def get_agent_timeout(default: float) -> float:
    """读取全局 Agent 总超时（秒）。"""
    value = get_env_float("PCE_AGENT_TIMEOUT")
    if value is None or value <= 0:
        return float(default)
    return float(value)


def get_completion_retries_per_model(default: int = 3) -> int:
    """读取每个模型的 completion 重试次数。"""
    value = get_env_int("PCE_COMPLETION_RETRIES_PER_MODEL")
    if value is None or value < 1:
        return int(default)
    return int(value)


def get_temperature(
    *,
    specific_key: str | None,
    default: float,
    fallback_key: str = "PCE_TEMPERATURE",
) -> float:
    """读取温度配置。

    优先级：
    1. fallback_key（全局总控）
    2. specific_key
    3. 代码默认值
    """
    if (fallback := get_env_float(fallback_key)) is not None:
        return fallback
    if specific_key:
        if (specific := get_env_float(specific_key)) is not None:
            return specific
    return float(default)


def configure_litellm_runtime() -> None:
    """尽可能关闭 litellm 对 stdout/stderr 的调试污染。

    PCE 的 MCP 服务运行在 stdio 模式下，任何第三方库向 stdout 输出非协议内容
    都可能破坏 MCP 传输。这里集中关闭 litellm 的调试输出与 provider 提示噪音。

    说明：
    - 仅做 best-effort 配置，不因 litellm 版本差异抛异常
    - 该函数应在服务入口和本地调试脚本启动时调用一次
    """
    try:
        import litellm

        # 关闭 provider 识别失败时的 stdout 提示
        litellm.suppress_debug_info = True
        # 关闭遗留 print_verbose 路径
        litellm.set_verbose = False

        try:
            from litellm._logging import _disable_debugging

            _disable_debugging()
        except Exception:
            pass

        for logger_name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
            try:
                logger = __import__("logging").getLogger(logger_name)
                logger.disabled = True
            except Exception:
                pass
    except Exception:
        # litellm 不可用或内部接口变化时，不影响主流程
        pass


def get_system_prompt_soft_limit() -> int:
    """根据 PCE_CONTEXT_WINDOW 计算动态注入块的 token 软上限。

    软上限 = context_window / 10，仅针对可压缩的动态注入块
    （structure.md + annotations/index.md + InsightCache），
    不含 SYSTEM_PROMPT_HEADER 和工具 schema。

    未配置时默认 context_window=200000，对应软上限 20000 token。
    """
    raw = get_env_text("PCE_CONTEXT_WINDOW")
    if raw is not None:
        try:
            context_window = int(raw)
            if context_window > 0:
                return max(1000, context_window // 10)
        except ValueError:
            pass
    return 20000  # 默认：200k context window / 10


def build_litellm_model(provider: str | None, model: str) -> str:
    """将 provider + model 机械拼装为 LiteLLM model 字符串。

    约定：
    - provider 表示真正的 LiteLLM provider，例如 `openrouter` / `openai` / `anthropic`
    - model 表示该 provider 下的模型名；对 openrouter 而言，模型名本身可以包含斜杠，
      例如 `openai/gpt-5`

    为兼容少量直接传入完整 LiteLLM model 的调用，若 model 已带同 provider 前缀，
    则原样返回，避免重复拼接。
    """
    text = model.strip()
    if not text:
        raise ValueError("model 不能为空")

    normalized_provider = provider.strip() if provider else ""
    if not normalized_provider:
        return text
    if text.startswith(f"{normalized_provider}/"):
        return text
    return f"{normalized_provider}/{text}"
