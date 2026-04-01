from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pce_v2 import PCEngine, InsightHostKind, InsightHostRef, ReconcileRequest
from pce_v2.navigation.builder import NavigationTreeBuilder


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, "utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_file(root / "backend" / "api.py", "def handler():\n    return 1\n")
        write_file(root / "frontend" / "app.ts", "export const app = 1;\n")

        engine = PCEngine()
        session = engine.bind(root)
        session.navigation_store.save(NavigationTreeBuilder().build(root))
        seeded = engine.seed_baselines(root)

        host = session.navigation_store.resolve_host_for_files(["backend/api.py"])
        session.insight_store.append(
            host,
            content="后端 api handler 负责基础响应。",
            files=["backend/api.py"],
        )
        before = len(session.insight_store.list_records(host))

        write_file(root / "backend" / "api.py", "# comment\n\ndef handler():\n    return 2\n")
        result = engine.reconcile(root, ReconcileRequest(dirty_files=["backend/api.py"]))
        after = len(session.insight_store.list_records(host))

        print(
            json.dumps(
                {
                    "seeded": seeded,
                    "host": host.model_dump(mode="json"),
                    "before": before,
                    "after": after,
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
