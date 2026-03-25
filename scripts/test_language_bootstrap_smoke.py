"""
本地语言自愈烟雾测试。

目标：
1. 在临时复制的仓库上验证 Serena 语言配置可被自动补齐；
2. 验证 `.vue/.js/.py` 文件能否在本地 Serena 通道下获得符号概览；
3. 可选追加一轮 PCEContext.handle_init 集成验证。

运行：
    uv run python scripts/test_language_bootstrap_smoke.py
    uv run python scripts/test_language_bootstrap_smoke.py --run-init
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pce._env import configure_litellm_runtime
from pce.serena_client import SerenaClient
from pce.serena_language_health import (
    preflight_serena_language_health,
    verify_serena_language_health,
)
from pce.server import PCEContext


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pce_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ignore_copy(_src: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".pce",
        "temp",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        "dist",
        "build",
    }
    return {name for name in names if name in ignored}


def _extract_symbol_summary(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= 220:
        return text
    return text[:220] + "..."


async def _cleanup(ctx: PCEContext | None) -> None:
    if ctx is None:
        return
    if ctx.watcher is not None:
        try:
            await ctx.watcher.stop()
        except Exception:
            pass


async def _run_smoke(run_init: bool) -> dict[str, Any]:
    load_dotenv(_pce_root() / ".env")
    configure_litellm_runtime()

    source_root = _repo_root()
    with tempfile.TemporaryDirectory(prefix="pce-language-smoke-") as tmp:
        copied_root = Path(tmp) / source_root.name
        shutil.copytree(source_root, copied_root, ignore=_ignore_copy)

        preflight = await preflight_serena_language_health(copied_root)
        probe_results: dict[str, Any] = {}
        verified = preflight

        async with SerenaClient.create(copied_root, timeout_seconds=300) as client:
            verified = await verify_serena_language_health(preflight, client)
            for rel_path in (
                "frontend/src/App.vue",
                "frontend/src/components/ControlPanel.vue",
                "frontend/src/main.js",
                "backend/app.py",
            ):
                try:
                    raw = await client.get_symbols_overview(rel_path, depth=1)
                    probe_results[rel_path] = {
                        "ok": True,
                        "summary": _extract_symbol_summary(raw),
                    }
                except Exception as exc:  # noqa: BLE001
                    probe_results[rel_path] = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        init_result: dict[str, Any] | None = None
        init_elapsed_s: float | None = None
        if run_init:
            ctx: PCEContext | None = None
            try:
                ctx = PCEContext()
                t0 = time.perf_counter()
                init_result = await ctx.handle_init(str(copied_root))
                init_elapsed_s = round(time.perf_counter() - t0, 2)
            finally:
                await _cleanup(ctx)

        return {
            "copied_root": str(copied_root),
            "preflight": preflight.model_dump(mode="json"),
            "verified": verified.model_dump(mode="json"),
            "probe_results": probe_results,
            "init_elapsed_s": init_elapsed_s,
            "init_result": init_result,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-init", action="store_true", help="额外跑一轮 PCEContext.handle_init")
    args = parser.parse_args()
    result = asyncio.run(_run_smoke(run_init=args.run_init))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
