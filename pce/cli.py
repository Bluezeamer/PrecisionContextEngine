"""PCE 命令行入口。"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pce",
        description="PCE (Precision Context Engine) — 代码库推理中间层",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve 子命令：启动 MCP Server，项目路径由 agent 通过 pce_init 工具传入
    subparsers.add_parser("serve", help="启动 MCP Server (stdio 模式)")

    args = parser.parse_args()

    if args.command == "serve":
        from .server import serve
        asyncio.run(serve())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
