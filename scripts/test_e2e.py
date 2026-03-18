"""
PCE 端到端集成测试 — 进程内直连，跑通真实 Serena + 真实 LLM。

跳过 stdio MCP Server 层，直接构造 PCEContext 并调用 handle_query / handle_impact。
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
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

from pce.agent import PCEAgent
from pce.insight_cache import InsightCache
from pce.serena_client import SerenaClient
from pce.server import PCEContext

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
MAX_SECONDS = 120.0  # 单次推理时间上限
SERENA_TIMEOUT = 30  # Serena 启动/初始化超时

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


async def _cleanup(
    ctx: PCEContext | None, client: SerenaClient | None
) -> None:
    """容错清理：即使某步失败也尽量完成后续清理。"""
    if ctx is not None:
        try:
            await ctx.watcher.stop()
            logger.info("FileWatcher 已停止")
        except Exception as exc:
            logger.warning("FileWatcher 停止失败: %s", exc)
    if client is not None:
        try:
            await client.disconnect()
            logger.info("Serena 已断开")
        except Exception as exc:
            logger.warning("Serena 断开失败: %s", exc)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
async def main() -> None:
    root = _root()
    load_dotenv(root / ".env")

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(os.getenv("PCE_PROJECT_PATH", str(root))).resolve()
    serena_path = Path(os.getenv("SERENA_PATH", str(root / "serena"))).resolve()

    _banner("PCE 端到端集成测试")
    _pprint("配置", {
        "project_path": str(project_path),
        "serena_path": str(serena_path),
        "model": os.getenv("PCE_MODEL", "(default)"),
        "max_seconds": MAX_SECONDS,
        "openrouter_key": "已设置" if os.getenv("OPENROUTER_API_KEY") else "未设置",
    })

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY 未配置，请检查 .env")
    if not serena_path.exists():
        raise RuntimeError(f"Serena 目录不存在: {serena_path}")

    client: SerenaClient | None = None
    ctx: PCEContext | None = None
    results: list[dict[str, Any]] = []

    try:
        # ── 初始化 ──────────────────────────────────────────────
        t0 = time.perf_counter()
        client = SerenaClient(timeout_seconds=SERENA_TIMEOUT)
        await client.connect(project_path, serena_path)

        cache = InsightCache(project_path)
        agent = PCEAgent(max_seconds=MAX_SECONDS, insight_cache=cache)
        ctx = PCEContext(
            project_path=project_path,
            serena_path=serena_path,
            serena_client=client,
            agent=agent,
            insight_cache=cache,
        )
        await ctx.watcher.start()

        # 触发 bootstrap（与 serve() 中行为一致）
        await ctx._bootstrap()

        init_elapsed = round(time.perf_counter() - t0, 2)
        _pprint("初始化完成", {
            "elapsed_s": init_elapsed,
            "serena_connected": client.connected,
            "watcher_running": ctx.watcher.running,
            "bootstrap_warnings": list(ctx._bootstrap_warnings),
        })

        # ── 场景 1: pce_query ───────────────────────────────────
        query_sid = str(uuid.uuid4())
        results.append(await _run_case(
            "pce_query",
            lambda: ctx.handle_query(
                query="PCEAgent 的 ReAct 循环是如何处理 deliver 调用的？",
                session_id=query_sid,
            ),
        ))

        # ── 场景 2: pce_impact ──────────────────────────────────
        impact_sid = str(uuid.uuid4())
        results.append(await _run_case(
            "pce_impact",
            lambda: ctx.handle_impact(
                target="SerenaClient",
                change_type="modify",
                session_id=impact_sid,
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
        await _cleanup(ctx, client)


if __name__ == "__main__":
    asyncio.run(main())
