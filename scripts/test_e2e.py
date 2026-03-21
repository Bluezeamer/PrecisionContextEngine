"""
PCE 端到端集成测试 — 进程内直连，跑通真实 Serena + 真实 LLM。

跳过 stdio MCP Server 层，直接构造 PCEContext 并调用 handle_init / handle_query / handle_impact。
即使单个场景失败，也会继续后续场景。

运行：
    uv run python scripts/test_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

from pce.server import PCEContext
from pce._env import configure_litellm_runtime, get_base_url, get_completion_overrides, get_env_text

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MAX_SECONDS = 120.0  # 单次推理时间上限（当前仅用于配置展示）

logger = logging.getLogger("pce.e2e")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {title}\n{sep}")


def _pprint(label: str, data: Any) -> None:
    print(f"\n[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


async def _run_case(
    name: str, func: Callable[[], Awaitable[dict[str, Any]]]
) -> dict[str, Any]:
    """执行单个测试场景，捕获异常并返回结构化结果。"""
    t0 = time.perf_counter()
    try:
        result = await func()
        elapsed = round(time.perf_counter() - t0, 2)
        payload = {"case": name, "ok": True, "elapsed_s": elapsed, "result": result}
        _pprint(f"{name} — 成功 ({elapsed}s)", payload)
        return payload
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 2)
        payload = {
            "case": name,
            "ok": False,
            "elapsed_s": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _pprint(f"{name} — 失败 ({elapsed}s)", payload)
        return payload


async def _cleanup(ctx: PCEContext | None) -> None:
    """容错清理：即使某步失败也尽量完成后续清理。"""
    if ctx is None:
        return
    if ctx.watcher is not None:
        try:
            await ctx.watcher.stop()
            logger.info("FileWatcher 已停止")
        except Exception as exc:
            logger.warning("FileWatcher 停止失败: %s", exc)
    serena_client = getattr(ctx, "serena_client", None)
    if serena_client is not None:
        try:
            await serena_client.disconnect()
            logger.info("Serena 已断开")
        except Exception as exc:
            logger.warning("Serena 断开失败: %s", exc)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main() -> None:
    root = _root()
    load_dotenv(root / ".env")
    configure_litellm_runtime()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or root).resolve()

    _banner("PCE 端到端集成测试")
    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()
    _pprint("配置", {
        "project_path": str(project_path),
        "provider": provider or "(missing)",
        "model": model or "(missing)",
        "max_seconds": MAX_SECONDS,
        "api_key": "已设置" if overrides.get("api_key") else "未设置",
        "base_url": get_base_url() or "(default)",
    })

    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置，请检查 .env")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置，请检查 .env")

    ctx: PCEContext | None = None
    results: list[dict[str, Any]] = []

    try:
        # ── 初始化（通过 pce_init 驱动，与 serve() 中 agent 调用行为一致）──
        t0 = time.perf_counter()
        ctx = PCEContext()
        init_result = await ctx.handle_init(str(project_path))

        init_elapsed = round(time.perf_counter() - t0, 2)
        _pprint("初始化完成", {
            "elapsed_s": init_elapsed,
            "init_result": init_result,
        })

        if not init_result.get("initialized"):
            raise RuntimeError(f"初始化失败: {init_result.get('error', '未知原因')}")

        # ── 场景 1: pce_query ───────────────────────────────────
        results.append(await _run_case(
            "pce_query",
            lambda: ctx.handle_query(
                query="PCEAgent 的 ReAct 循环是如何处理 deliver 调用的？",
            ),
        ))

        # ── 场景 2: pce_impact ──────────────────────────────────
        results.append(await _run_case(
            "pce_impact",
            lambda: ctx.handle_impact(
                target="SerenaClient",
                change_type="modify",
                file=None,
            ),
        ))

        # ── 汇总 ────────────────────────────────────────────────
        ok_count = sum(1 for r in results if r["ok"])
        _banner("测试汇总")
        _pprint("结果", {
            "total": len(results),
            "passed": ok_count,
            "failed": len(results) - ok_count,
            "cases": [
                {"case": r["case"], "ok": r["ok"], "elapsed_s": r["elapsed_s"]}
                for r in results
            ],
        })
    finally:
        await _cleanup(ctx)


if __name__ == "__main__":
    asyncio.run(main())
