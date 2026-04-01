from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pce_v2 import PCEngine, QueryRequest
from pce_v2.runtime.react import MinimalReActRuntime


class FakeRuntime(MinimalReActRuntime):
    def __init__(self) -> None:
        super().__init__()
        self._step = 0

    async def _complete(self, messages, tools):  # type: ignore[override]
        del messages, tools
        if self._step == 0:
            self._step += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_nav",
                                    "function": {
                                        "name": "navigation_read",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_deliver",
                                "function": {
                                    "name": "deliver",
                                    "arguments": json.dumps(
                                        {"answer": "## 结论\ntrace smoke", "confidence": "high"},
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    engine = PCEngine()
    session = engine.bind(project_root)
    runtime = FakeRuntime()
    result = await runtime.run_query(session, QueryRequest(question="trace smoke"))
    trace_path = session.trace_store.trace_path(result["request_id"])
    print(json.dumps({"result": result, "trace_path": str(trace_path)}, ensure_ascii=False, indent=2))
    print("--- TRACE ---")
    print(trace_path.read_text("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
