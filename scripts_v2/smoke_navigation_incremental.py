from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pce_v2 import PCEngine, ReconcileRequest


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, "utf-8")


def run_patch_case() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_file(root / "backend" / "api.py", "def api():\n    return 1\n")
        write_file(root / "backend" / "service" / "logic.py", "def logic():\n    return 1\n")
        write_file(root / "frontend" / "app.ts", "export const app = 1\n")

        engine = PCEngine()
        engine.seed_baselines(root)
        write_file(root / "backend" / "service" / "logic.py", "def logic():\n    return 2\n")
        result = engine.reconcile(root, ReconcileRequest(dirty_files=["backend/service/logic.py"]))
        return result.model_dump(mode="json")


def run_rebuild_case() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_file(root / "backend" / "api.py", "def api():\n    return 1\n")
        write_file(root / "frontend" / "app.ts", "export const app = 1\n")

        engine = PCEngine()
        engine.seed_baselines(root)
        write_file(root / "root_level.py", "value = 1\n")
        result = engine.reconcile(root, ReconcileRequest(dirty_files=["root_level.py"]))
        return result.model_dump(mode="json")


def main() -> None:
    print(
        json.dumps(
            {
                "patch_case": run_patch_case(),
                "rebuild_case": run_rebuild_case(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
