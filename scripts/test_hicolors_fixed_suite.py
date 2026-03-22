"""
hicolors 固定题集回归脚本。

目标：
1. 用固定 query / impact 题集评估 PCE 当前输出质量；
2. 避免后续继续依赖临时构造问题做主观比较；
3. 将结果写入项目根目录 temp/，便于与 ACE 陪跑结果并排查看。

运行：
    uv run python scripts/test_hicolors_fixed_suite.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

from pce._env import (
    configure_litellm_runtime,
    get_base_url,
    get_completion_overrides,
    get_env_text,
)
from pce.server import PCEContext

logger = logging.getLogger("pce.hicolors.fixed_suite")


FIXED_CASES: list[dict[str, Any]] = [
    {
        "name": "query_layer_order_mainline",
        "kind": "query",
        "payload": {
            "query": "hicolors 项目中，从前端发起“生成层序”到后端调用核心算法 auto_generate_layer_order 的主干链路是什么？请给出入口文件、关键函数、状态传递和文件位置。"
        },
    },
    {
        "name": "query_height_map_mainline",
        "kind": "query",
        "payload": {
            "query": "hicolors 项目中，从前端点击“生成灰度图”到后端高度图生成完成的主干链路是什么？请给出入口文件、关键函数、状态传递和文件位置。"
        },
    },
    {
        "name": "impact_auto_generate_layer_order_signature",
        "kind": "impact",
        "payload": {
            "target": "auto_generate_layer_order",
            "change_type": "change_signature",
            "file": "hicolors_logic_v2/layer_order.py",
        },
    },
    {
        "name": "impact_layer_order_response_field",
        "kind": "impact",
        "payload": {
            "target": "backend/app.py 中 /api/v2/layer-order 返回字段 layer_order",
            "change_type": "modify",
            "file": "backend/app.py",
        },
    },
    {
        "name": "impact_compute_height_and_grayscale_signature",
        "kind": "impact",
        "payload": {
            "target": "compute_height_and_grayscale",
            "change_type": "change_signature",
            "file": "hicolors_logic_v2/height_mapping.py",
        },
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pce_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {title}\n{sep}")


def _pprint(label: str, data: Any) -> None:
    print(f"\n[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


async def _run_case(
    name: str,
    func: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
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


async def main() -> None:
    repo_root = _repo_root()
    pce_root = _pce_root()
    load_dotenv(pce_root / ".env")
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or repo_root).resolve()
    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()
    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置，请检查 .env")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置，请检查 .env")

    _banner("hicolors 固定题集测试")
    _pprint(
        "配置",
        {
            "project_path": str(project_path),
            "provider": provider,
            "model": model,
            "api_key": "已设置" if overrides.get("api_key") else "未设置",
            "base_url": get_base_url() or "(default)",
            "cases": [case["name"] for case in FIXED_CASES],
        },
    )

    ctx: PCEContext | None = None
    results: list[dict[str, Any]] = []

    try:
        ctx = PCEContext()
        t0 = time.perf_counter()
        init_result = await ctx.handle_init(str(project_path))
        init_elapsed = round(time.perf_counter() - t0, 2)
        _pprint(
            "初始化完成",
            {
                "elapsed_s": init_elapsed,
                "init_result": init_result,
            },
        )
        if not init_result.get("initialized"):
            raise RuntimeError(f"初始化失败: {init_result.get('error', '未知原因')}")

        for case in FIXED_CASES:
            if case["kind"] == "query":
                results.append(
                    await _run_case(
                        case["name"],
                        lambda payload=case["payload"]: ctx.handle_query(**payload),
                    )
                )
            else:
                results.append(
                    await _run_case(
                        case["name"],
                        lambda payload=case["payload"]: ctx.handle_impact(**payload),
                    )
                )

        summary = {
            "project_path": str(project_path),
            "init_elapsed_s": init_elapsed,
            "passed": sum(1 for item in results if item["ok"]),
            "failed": sum(1 for item in results if not item["ok"]),
            "results": results,
        }
        out_path = repo_root / "temp" / "pce_hicolors_fixed_suite.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        _banner("测试汇总")
        _pprint(
            "结果",
            {
                "output_file": str(out_path),
                "passed": summary["passed"],
                "failed": summary["failed"],
                "cases": [
                    {
                        "case": item["case"],
                        "ok": item["ok"],
                        "elapsed_s": item["elapsed_s"],
                    }
                    for item in results
                ],
            },
        )
    finally:
        await _cleanup(ctx)


if __name__ == "__main__":
    asyncio.run(main())
