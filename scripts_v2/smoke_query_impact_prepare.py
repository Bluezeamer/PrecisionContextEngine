from __future__ import annotations

import json
from pathlib import Path

from pce_v2 import ImpactRequest, PCEngine, QueryRequest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    engine = PCEngine()
    query = engine.prepare_query_execution(
        project_root,
        QueryRequest(question="PCE v2 的 query 主线要如何执行？"),
    )
    impact = engine.prepare_impact_execution(
        project_root,
        ImpactRequest(target="NavigationTreeBuilder", change_type="modify"),
    )
    print(
        json.dumps(
            {
                "query_mode": query["mode"],
                "impact_mode": impact["mode"],
                "query_tools": [tool["name"] for tool in query["tools"]],
                "impact_tools": [tool["name"] for tool in impact["tools"]],
                "query_context_blocks": len(query["context_blocks"]),
                "impact_context_blocks": len(impact["context_blocks"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
