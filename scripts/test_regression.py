"""
PCE 迁移后最小回归验证 — init → query → impact → sync → digest smoke。

从零开始（.pce 已删除），验证全链路功能完整性并记录每步执行时间。

运行：
    uv run python scripts/test_regression.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pce._env import configure_litellm_runtime, get_base_url, get_completion_overrides, get_env_text
from pce.server import PCEContext

logger = logging.getLogger("pce.regression")

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {title}\n{sep}")


def _pprint(label: str, data: Any) -> None:
    print(f"\n[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _quality_check(label: str, condition: bool, detail: str = "") -> dict[str, Any]:
    """记录一个质量断言，不中断流程。"""
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {label}" + (f" — {detail}" if detail else ""))
    return {"check": label, "ok": condition, "detail": detail}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main() -> None:
    root = _root()
    load_dotenv(root / ".env")
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or root).resolve()

    _banner("PCE 迁移后回归验证")
    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()
    _pprint("配置", {
        "project_path": str(project_path),
        "provider": provider or "(missing)",
        "model": model or "(missing)",
        "api_key": "已设置" if overrides.get("api_key") else "未设置",
        "base_url": get_base_url() or "(default)",
    })

    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置")

    ctx: PCEContext | None = None
    timings: dict[str, float] = {}
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        # ================================================================
        # 1. pce_init（从零开始，.pce 已删除）
        # ================================================================
        _banner("1. pce_init")
        t0 = time.perf_counter()
        ctx = PCEContext()
        init_result = await ctx.handle_init(str(project_path))
        elapsed = round(time.perf_counter() - t0, 2)
        timings["init"] = elapsed

        _pprint(f"init 完成 ({elapsed}s)", {
            "initialized": init_result.get("initialized"),
            "files_indexed": init_result.get("files_indexed"),
            "annotations_generated": init_result.get("annotations_generated"),
        })

        ok = bool(init_result.get("initialized"))
        checks.append(_quality_check("init 成功", ok))
        checks.append(_quality_check(
            "init 时间 < 300s", elapsed < 300,
            f"{elapsed}s",
        ))

        pce_dir = project_path / ".pce"
        checks.append(_quality_check(
            "structure.md 已生成",
            (pce_dir / "structure.md").exists(),
        ))
        checks.append(_quality_check(
            "annotations/index.md 已生成",
            (pce_dir / "annotations" / "index.md").exists(),
        ))

        modules_dir = pce_dir / "annotations" / "modules"
        module_count = len(list(modules_dir.glob("*.md"))) if modules_dir.exists() else 0
        checks.append(_quality_check(
            "模块文档 > 0",
            module_count > 0,
            f"{module_count} 个模块文档",
        ))

        if not ok:
            raise RuntimeError(f"init 失败: {init_result.get('error', '未知')}")

        # ================================================================
        # 2. pce_query
        # ================================================================
        _banner("2. pce_query")
        t0 = time.perf_counter()
        query_result = await ctx.handle_query(
            query="PCEAgent 的 query 方法是如何构建 system prompt 的？"
        )
        elapsed = round(time.perf_counter() - t0, 2)
        timings["query"] = elapsed

        answer = query_result.get("answer") or query_result.get("markdown", "")
        confidence = query_result.get("confidence")
        _pprint(f"query 完成 ({elapsed}s)", {
            "answer_length": len(answer),
            "confidence": confidence,
            "answer_preview": answer[:500] if answer else "(empty)",
        })

        checks.append(_quality_check("query 有返回", bool(answer)))
        checks.append(_quality_check(
            "query answer 非异常兜底",
            not answer.startswith("__REACT_"),
            answer[:80] if answer.startswith("__REACT_") else "",
        ))
        checks.append(_quality_check(
            "query 时间 < 120s", elapsed < 120,
            f"{elapsed}s",
        ))

        # ================================================================
        # 3. pce_impact
        # ================================================================
        _banner("3. pce_impact")
        t0 = time.perf_counter()
        impact_result = await ctx.handle_impact(
            target="SerenaClient",
            change_type="modify",
            file=None,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        timings["impact"] = elapsed

        impact_answer = impact_result.get("answer") or impact_result.get("markdown", "")
        _pprint(f"impact 完成 ({elapsed}s)", {
            "answer_length": len(impact_answer),
            "confidence": impact_result.get("confidence"),
            "answer_preview": impact_answer[:500] if impact_answer else "(empty)",
        })

        checks.append(_quality_check("impact 有返回", bool(impact_answer)))
        checks.append(_quality_check(
            "impact answer 非异常兜底",
            not impact_answer.startswith("__REACT_"),
            impact_answer[:80] if impact_answer.startswith("__REACT_") else "",
        ))
        checks.append(_quality_check(
            "impact 时间 < 120s", elapsed < 120,
            f"{elapsed}s",
        ))

        # ================================================================
        # 4. pce_sync（无实际变更，验证不报错）
        # ================================================================
        _banner("4. pce_sync")
        t0 = time.perf_counter()
        sync_result = await ctx.handle_sync()
        elapsed = round(time.perf_counter() - t0, 2)
        timings["sync"] = elapsed

        _pprint(f"sync 完成 ({elapsed}s)", sync_result)
        checks.append(_quality_check("sync 不报错", True))
        checks.append(_quality_check(
            "sync 时间 < 60s", elapsed < 60,
            f"{elapsed}s",
        ))

        # ================================================================
        # 5. digest smoke（验证 DigestPlanner 能跑通）
        # ================================================================
        _banner("5. digest smoke")
        t0 = time.perf_counter()
        try:
            from pce.digest_agent import DigestPlanner
            from pce.insight_cache import InsightCache
            from pce.staging import StagingArea

            staging = StagingArea(project_path)
            insight_cache = InsightCache(project_path)
            dirty = await staging.list_pending_reindex()
            planner = DigestPlanner(
                project_root=project_path,
                insight_cache=insight_cache,
            )
            task_list = await planner.build(dirty)
            elapsed = round(time.perf_counter() - t0, 2)
            timings["digest_plan"] = elapsed

            _pprint(f"digest plan 完成 ({elapsed}s)", {
                "dirty_changed": len(dirty.changed),
                "dirty_deleted": len(dirty.deleted),
                "task_count": len(task_list.items),
                "warnings": list(task_list.warnings)[:5],
            })
            checks.append(_quality_check("digest planner 不报错", True))
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            timings["digest_plan"] = elapsed
            checks.append(_quality_check(
                "digest planner 不报错", False, str(exc),
            ))
            errors.append({"stage": "digest", "error": traceback.format_exc()})

    except Exception as exc:
        errors.append({"stage": "main", "error": traceback.format_exc()})
        print(f"\n!!! 主流程异常: {exc}")

    finally:
        # 清理
        if ctx is not None:
            if ctx.watcher is not None:
                try:
                    await ctx.watcher.stop()
                except Exception:
                    pass
            serena_client = getattr(ctx, "serena_client", None)
            if serena_client is not None:
                try:
                    await serena_client.disconnect()
                except Exception:
                    pass

    # ================================================================
    # 汇总
    # ================================================================
    _banner("回归验证汇总")

    passed = sum(1 for c in checks if c["ok"])
    failed = sum(1 for c in checks if not c["ok"])

    _pprint("执行时间", timings)
    total_time = sum(timings.values())
    print(f"\n  总耗时: {total_time:.1f}s")

    _pprint("质量检查", {
        "total": len(checks),
        "passed": passed,
        "failed": failed,
        "failures": [c for c in checks if not c["ok"]],
    })

    if errors:
        _pprint("错误详情", errors)

    # 最终判定
    print(f"\n{'='*80}")
    if failed == 0 and not errors:
        print("  结论: 全部通过 ✓")
    else:
        print(f"  结论: {failed} 项失败, {len(errors)} 个异常")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
