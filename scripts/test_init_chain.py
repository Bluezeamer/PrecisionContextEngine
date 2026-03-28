"""
PCE init-only 链路测试脚本。

目标：
1. 只测试 init，不混入 query / impact；
2. 支持两种模式：
   - `mcp-stdio`：通过 `uv run python -m pce.cli serve` 拉起本地 MCP 服务，再调用 `pce_init`
   - `direct`：进程内直接构造 `PCEContext` 并调用 `handle_init`
3. 显式打印 project_path、`.pce` 路径、耗时拆分，避免默认路径造成误判。

示例：
    uv run python scripts/test_init_chain.py --mode mcp-stdio --project-path /abs/path --clean-pce
    uv run python scripts/test_init_chain.py --mode direct --project-path /abs/path --clean-pce
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from pce._env import (
    configure_litellm_runtime,
    get_base_url,
    get_completion_overrides,
    get_env_text,
)
from pce.server import PCEContext

logger = logging.getLogger("pce.init_chain")


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {title}\n{sep}")


def _pprint(label: str, data: Any) -> None:
    print(f"\n[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _normalize_tool_result(value: Any) -> Any:
    if hasattr(value, "structuredContent") and getattr(value, "structuredContent") is not None:
        return getattr(value, "structuredContent")
    if hasattr(value, "content"):
        content = getattr(value, "content")
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return text
        return content
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _list_tools_count(tools_result: Any) -> int | None:
    if hasattr(tools_result, "tools"):
        tools = getattr(tools_result, "tools")
        if isinstance(tools, list):
            return len(tools)
    if isinstance(tools_result, dict):
        tools = tools_result.get("tools")
        if isinstance(tools, list):
            return len(tools)
    return None


async def _cleanup_direct(ctx: PCEContext | None) -> None:
    if ctx is None:
        return
    if ctx.watcher is not None:
        try:
            await ctx.watcher.stop()
        except Exception:
            logger.exception("停止 FileWatcher 失败")
    serena_client = getattr(ctx, "serena_client", None)
    if serena_client is not None:
        try:
            await serena_client.disconnect()
        except Exception:
            logger.exception("断开 Serena 失败")


async def _run_direct(project_path: Path) -> dict[str, Any]:
    ctx: PCEContext | None = None
    t0 = time.perf_counter()
    try:
        ctx = PCEContext()
        t1 = time.perf_counter()
        result = await ctx.handle_init(str(project_path))
        t2 = time.perf_counter()
        return {
            "mode": "direct",
            "timing": {
                "construct_context_s": round(t1 - t0, 2),
                "handle_init_s": round(t2 - t1, 2),
                "total_s": round(t2 - t0, 2),
            },
            "result": result,
        }
    finally:
        await _cleanup_direct(ctx)


async def _run_mcp_stdio(project_path: Path, root: Path) -> dict[str, Any]:
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "pce.cli", "serve"],
        env=os.environ.copy(),
        cwd=root,
    )

    t0 = time.perf_counter()
    async with stdio_client(server_params) as (read, write):
        t1 = time.perf_counter()
        async with ClientSession(read, write) as session:
            t2 = time.perf_counter()
            await session.initialize()
            t3 = time.perf_counter()
            tools = await session.list_tools()
            t4 = time.perf_counter()
            init_result = await session.call_tool(
                "pce_init",
                {"project_path": str(project_path)},
            )
            t5 = time.perf_counter()

    return {
        "mode": "mcp-stdio",
        "timing": {
            "spawn_stdio_s": round(t1 - t0, 2),
            "create_session_s": round(t2 - t1, 2),
            "initialize_s": round(t3 - t2, 2),
            "list_tools_s": round(t4 - t3, 2),
            "pce_init_call_s": round(t5 - t4, 2),
            "total_s": round(t5 - t0, 2),
        },
        "tools_count": _list_tools_count(tools),
        "result": _normalize_tool_result(init_result),
    }


def _resolve_project_path(args: argparse.Namespace, root: Path) -> Path:
    if args.project_path:
        return Path(args.project_path).expanduser().resolve()
    env_path = get_env_text("PCE_PROJECT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只测试 PCE init 链路，不混入 query / impact。",
    )
    parser.add_argument(
        "--mode",
        choices=("mcp-stdio", "direct"),
        default="mcp-stdio",
        help="测试模式：默认走本地 stdio MCP 服务。",
    )
    parser.add_argument(
        "--project-path",
        help="待初始化的项目根路径；不传则回退到 PCE_PROJECT_PATH，再回退到当前 PCE 仓库根。",
    )
    parser.add_argument(
        "--clean-pce",
        action="store_true",
        help="执行前删除目标项目下的 .pce 目录。",
    )
    return parser


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = _root()
    load_dotenv(root / ".env", override=False)
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()
    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置")

    project_path = _resolve_project_path(args, root)
    pce_path = project_path / ".pce"

    if args.clean_pce and pce_path.exists():
        shutil.rmtree(pce_path)

    _banner("PCE init-only 链路测试")
    _pprint("配置", {
        "mode": args.mode,
        "script_root": str(root),
        "cwd": os.getcwd(),
        "project_path": str(project_path),
        "pce_path": str(pce_path),
        "pce_exists_before": pce_path.exists(),
        "provider": provider,
        "model": model,
        "api_key": "已设置" if overrides.get("api_key") else "未设置",
        "base_url": get_base_url() or "(default)",
        "clean_pce": bool(args.clean_pce),
    })

    if not project_path.exists():
        raise RuntimeError(f"project_path 不存在: {project_path}")
    if not project_path.is_dir():
        raise RuntimeError(f"project_path 不是目录: {project_path}")

    if args.mode == "mcp-stdio":
        payload = await _run_mcp_stdio(project_path, root)
    else:
        payload = await _run_direct(project_path)

    payload["project_path"] = str(project_path)
    payload["pce_path"] = str(pce_path)
    payload["pce_exists_after"] = pce_path.exists()

    _pprint("结果", payload)


if __name__ == "__main__":
    asyncio.run(main())
