"""
PCE vs ACE 对比测试脚本。

目标：
1. 以固定题集跑 2 个 query + 2 个 impact；
2. 将 PCE 的 query/impact 推理时限临时放宽到 20 分钟；
3. 统计 init 各阶段耗时，以及每个请求内部 Serena 工具调用耗时；
4. 输出结构化 JSON 结果，便于和 ACE 对照整理。

运行：
    uv run python scripts/test_pce_ace_compare.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pce._env import (
    configure_litellm_runtime,
    get_base_url,
    get_completion_overrides,
    get_env_text,
)
from pce.agent import PCEAgent
from pce.server import PCEContext

logger = logging.getLogger("pce.compare")

QUERY_CASES: list[dict[str, str]] = [
    {
        "name": "query_layer_order_mainline",
        "question": (
            "hicolors 项目中，从前端发起“生成层序”到后端调用核心算法 "
            "auto_generate_layer_order 的主干链路是什么？"
            "请给出入口文件、关键函数、状态传递和文件位置。"
        ),
    },
    {
        "name": "query_height_map_mainline",
        "question": (
            "hicolors 项目中，从前端点击“生成灰度图”到后端高度图生成完成的主干链路是什么？"
            "请给出入口文件、关键函数、状态传递和文件位置。"
        ),
    },
]

IMPACT_CASES: list[dict[str, str]] = [
    {
        "name": "impact_auto_generate_layer_order_signature",
        "target": "auto_generate_layer_order",
        "change_type": "change_signature",
        "file": "hicolors_logic_v2/layer_order.py",
    },
    {
        "name": "impact_layer_order_response_field",
        "target": "backend/app.py 中 /api/v2/layer-order 返回字段 layer_order",
        "change_type": "modify",
        "file": "backend/app.py",
    },
]

REQ_START_RE = re.compile(r'^\[req=(?P<req>[0-9a-f]+)\] (?P<kind>QUERY|IMPACT)\b')
REQ_TOOL_RE = re.compile(
    r'^\[req=(?P<req>[0-9a-f]+)\] round=(?P<round>\d+) -> (?P<preview>.+?) '
    r'(?P<elapsed>\d+\.\d+)s (?P<chars>\d+)chars$'
)
REQ_DELIVER_RE = re.compile(
    r'^\[req=(?P<req>[0-9a-f]+)\] DELIVER confidence=(?P<confidence>\w+) '
    r'elapsed=(?P<elapsed>\d+\.\d+)s rounds=(?P<rounds>\d+)$'
)
BOOTSTRAP_STAGE_RE = re.compile(
    r"^Bootstrap 阶段耗时: (?P<stage>[a-z_]+)=(?P<elapsed>\d+\.\d+)s"
)
BOOTSTRAP_TOTAL_RE = re.compile(r"^Bootstrap 总耗时: (?P<elapsed>\d+\.\d+)s$")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pce_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _tool_name_from_preview(preview: str) -> str:
    return preview.split(" ", 1)[0] if preview else "unknown"


@dataclass
class ToolCallRecord:
    round_num: int
    preview: str
    tool_name: str
    elapsed_s: float
    chars: int


@dataclass
class RequestRecord:
    req_id: str
    kind: str
    started_at: float = field(default_factory=time.monotonic)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    delivered: bool = False
    deliver_elapsed_s: float | None = None
    deliver_rounds: int | None = None
    confidence: str | None = None

    def summary(self) -> dict[str, Any]:
        by_tool: dict[str, dict[str, Any]] = {}
        for item in self.tool_calls:
            bucket = by_tool.setdefault(
                item.tool_name,
                {"calls": 0, "total_elapsed_s": 0.0, "max_elapsed_s": 0.0},
            )
            bucket["calls"] += 1
            bucket["total_elapsed_s"] = round(bucket["total_elapsed_s"] + item.elapsed_s, 2)
            bucket["max_elapsed_s"] = round(max(bucket["max_elapsed_s"], item.elapsed_s), 2)
        return {
            "req_id": self.req_id,
            "kind": self.kind,
            "tool_calls_count": len(self.tool_calls),
            "tool_calls": [
                {
                    "round": item.round_num,
                    "preview": item.preview,
                    "tool_name": item.tool_name,
                    "elapsed_s": round(item.elapsed_s, 2),
                    "chars": item.chars,
                }
                for item in self.tool_calls
            ],
            "tool_stats": by_tool,
            "delivered": self.delivered,
            "deliver_elapsed_s": self.deliver_elapsed_s,
            "deliver_rounds": self.deliver_rounds,
            "confidence": self.confidence,
        }


class TimingLogCollector(logging.Handler):
    """从现有日志中提取 bootstrap 和 req/tool 调用耗时。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.request_order: list[str] = []
        self.requests: dict[str, RequestRecord] = {}
        self.bootstrap_stages: list[dict[str, Any]] = []
        self.bootstrap_total_s: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()

        match = REQ_START_RE.match(message)
        if match:
            req_id = match.group("req")
            kind = match.group("kind").lower()
            self.requests[req_id] = RequestRecord(req_id=req_id, kind=kind)
            self.request_order.append(req_id)
            return

        match = REQ_TOOL_RE.match(message)
        if match:
            req_id = match.group("req")
            req = self.requests.get(req_id)
            if req is None:
                return
            preview = match.group("preview")
            req.tool_calls.append(
                ToolCallRecord(
                    round_num=int(match.group("round")),
                    preview=preview,
                    tool_name=_tool_name_from_preview(preview),
                    elapsed_s=float(match.group("elapsed")),
                    chars=int(match.group("chars")),
                )
            )
            return

        match = REQ_DELIVER_RE.match(message)
        if match:
            req_id = match.group("req")
            req = self.requests.get(req_id)
            if req is None:
                return
            req.delivered = True
            req.deliver_elapsed_s = round(float(match.group("elapsed")), 2)
            req.deliver_rounds = int(match.group("rounds"))
            req.confidence = match.group("confidence")
            return

        match = BOOTSTRAP_STAGE_RE.match(message)
        if match:
            self.bootstrap_stages.append(
                {
                    "stage": match.group("stage"),
                    "elapsed_s": round(float(match.group("elapsed")), 2),
                }
            )
            return

        match = BOOTSTRAP_TOTAL_RE.match(message)
        if match:
            self.bootstrap_total_s = round(float(match.group("elapsed")), 2)

    def latest_request(self) -> RequestRecord | None:
        if not self.request_order:
            return None
        return self.requests[self.request_order[-1]]


def _extract_answer(payload: dict[str, Any]) -> str:
    return str(payload.get("answer") or payload.get("markdown") or "")


async def _cleanup(ctx: PCEContext | None) -> None:
    if ctx is None:
        return
    if ctx.watcher is not None:
        try:
            await ctx.watcher.stop()
        except Exception as exc:  # pragma: no cover
            logger.warning("FileWatcher 停止失败: %s", exc)
    serena_client = getattr(ctx, "serena_client", None)
    if serena_client is not None:
        try:
            await serena_client.disconnect()
        except Exception as exc:  # pragma: no cover
            logger.warning("Serena 断开失败: %s", exc)


async def main() -> None:
    repo_root = _repo_root()
    pce_root = _pce_root()
    load_dotenv(pce_root / ".env")
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    collector = TimingLogCollector()
    root_logger = logging.getLogger()
    root_logger.addHandler(collector)

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or repo_root).resolve()
    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()
    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置，请检查 .env")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置，请检查 .env")

    ctx: PCEContext | None = None
    results: list[dict[str, Any]] = []
    timings: dict[str, Any] = {
        "init_total_s": None,
        "bootstrap_stages": [],
        "cases": [],
    }

    try:
        ctx = PCEContext()

        init_start = time.perf_counter()
        init_result = await ctx.handle_init(str(project_path))
        init_elapsed = round(time.perf_counter() - init_start, 2)
        timings["init_total_s"] = init_elapsed
        timings["bootstrap_stages"] = collector.bootstrap_stages
        timings["bootstrap_total_from_log_s"] = collector.bootstrap_total_s

        if not init_result.get("initialized"):
            raise RuntimeError(f"初始化失败: {init_result.get('error', '未知原因')}")

        if ctx.insight_cache is None:
            raise RuntimeError("InsightCache 未初始化")
        ctx.agent = PCEAgent(insight_cache=ctx.insight_cache, max_seconds=1200.0)

        for case in QUERY_CASES:
            started_reqs = len(collector.request_order)
            t0 = time.perf_counter()
            payload = await ctx.handle_query(query=case["question"])
            elapsed = round(time.perf_counter() - t0, 2)
            req = collector.latest_request() if len(collector.request_order) > started_reqs else None
            result = {
                "name": case["name"],
                "kind": "query",
                "elapsed_s": elapsed,
                "answer_length": len(_extract_answer(payload)),
                "confidence": payload.get("confidence"),
                "ok": bool(_extract_answer(payload)) and not _extract_answer(payload).startswith("__REACT_"),
                "result": payload,
                "request_trace": req.summary() if req else None,
            }
            results.append(result)
            timings["cases"].append(
                {
                    "name": case["name"],
                    "kind": "query",
                    "elapsed_s": elapsed,
                    "req_id": req.req_id if req else None,
                    "tool_stats": req.summary().get("tool_stats") if req else {},
                }
            )

        for case in IMPACT_CASES:
            started_reqs = len(collector.request_order)
            t0 = time.perf_counter()
            payload = await ctx.handle_impact(
                target=case["target"],
                change_type=case["change_type"],
                file=case["file"],
            )
            elapsed = round(time.perf_counter() - t0, 2)
            req = collector.latest_request() if len(collector.request_order) > started_reqs else None
            result = {
                "name": case["name"],
                "kind": "impact",
                "elapsed_s": elapsed,
                "answer_length": len(_extract_answer(payload)),
                "confidence": payload.get("confidence"),
                "ok": bool(_extract_answer(payload)) and not _extract_answer(payload).startswith("__REACT_"),
                "result": payload,
                "request_trace": req.summary() if req else None,
            }
            results.append(result)
            timings["cases"].append(
                {
                    "name": case["name"],
                    "kind": "impact",
                    "elapsed_s": elapsed,
                    "req_id": req.req_id if req else None,
                    "tool_stats": req.summary().get("tool_stats") if req else {},
                }
            )

        aggregate_tools: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "total_elapsed_s": 0.0, "max_elapsed_s": 0.0}
        )
        for item in results:
            trace = item.get("request_trace") or {}
            for tool_name, stats in (trace.get("tool_stats") or {}).items():
                bucket = aggregate_tools[tool_name]
                bucket["calls"] += int(stats["calls"])
                bucket["total_elapsed_s"] = round(
                    bucket["total_elapsed_s"] + float(stats["total_elapsed_s"]), 2
                )
                bucket["max_elapsed_s"] = round(
                    max(bucket["max_elapsed_s"], float(stats["max_elapsed_s"])), 2
                )

        summary = {
            "project_path": str(project_path),
            "provider": provider,
            "model": model,
            "base_url": get_base_url() or "(default)",
            "init_result": init_result,
            "timings": {
                **timings,
                "aggregate_tool_stats": dict(sorted(aggregate_tools.items())),
            },
            "results": results,
        }

        out_path = repo_root / "temp" / "pce_ace_compare_pce.json"
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n输出文件: {out_path}")
    finally:
        root_logger.removeHandler(collector)
        await _cleanup(ctx)


if __name__ == "__main__":
    asyncio.run(main())
