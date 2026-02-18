"""PCE 命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pce",
        description="PCE (Precision Context Engine) — 代码库推理中间层",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve 子命令
    serve_parser = subparsers.add_parser("serve", help="启动 MCP Server (stdio 模式)")
    serve_parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="目标项目根路径(默认使用 PCE_PROJECT_PATH 环境变量或当前目录)",
    )
    serve_parser.add_argument(
        "--serena",
        type=str,
        default=None,
        help="Serena 安装路径(默认使用 SERENA_PATH 环境变量或 ./serena)",
    )

    args = parser.parse_args()

    if args.command == "serve":
        if args.project:
            os.environ["PCE_PROJECT_PATH"] = str(Path(args.project).resolve())
        if args.serena:
            os.environ["SERENA_PATH"] = str(Path(args.serena).resolve())

        from .server import serve
        asyncio.run(serve())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
