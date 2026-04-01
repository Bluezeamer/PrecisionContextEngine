from __future__ import annotations

import json
from pathlib import Path

from pce_v2.engine import PCEngine


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    engine = PCEngine()
    session = engine.bind(project_root)
    tree = session.navigation_store.ensure_tree()
    print(
        json.dumps(
            {
                "root": str(tree.root_path),
                "nodes": len(tree.nodes),
                "rules": len(tree.rules),
                "bindings": len(tree.bindings),
                "tree_path": str(session.navigation_store.tree_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
