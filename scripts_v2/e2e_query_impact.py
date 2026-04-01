from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from pce_v2 import ImpactRequest, PCEngine, QueryRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v2 query/impact against a real model.")
    parser.add_argument("mode", choices=["query", "impact"])
    parser.add_argument("prompt", help="query question or impact target")
    parser.add_argument("--change-type", default="modify")
    parser.add_argument("--file", default=None)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(args.project).resolve()
    load_dotenv(project_root / ".env", override=False)

    engine = PCEngine()
    if args.mode == "query":
        result = await engine.run_query(project_root, QueryRequest(question=args.prompt))
    else:
        result = await engine.run_impact(
            project_root,
            ImpactRequest(target=args.prompt, change_type=args.change_type, file=args.file),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
