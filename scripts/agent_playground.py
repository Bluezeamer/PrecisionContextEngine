"""PCE Agent 独立测试环境。

脱离 MCP 服务层，支持两种模式快速迭代调试 agent 行为：
- mock  模式：使用 MockToolProvider，离线秒启动
- repl  模式：连接真实 Serena，一次启动成本后反复测试

用法示例：
  # mock 模式，使用默认规则
  uv run python scripts/agent_playground.py --mode mock --query "查找 PCEAgent 类的定义位置"

  # mock 模式，使用录制文件
  uv run python scripts/agent_playground.py --mode mock --recording temp/rec.json --query "..."

  # repl 模式，连接真实 Serena
  uv run python scripts/agent_playground.py --mode repl --project-path . --serena-path ../serena

  # repl 模式，同时录制交互
  uv run python scripts/agent_playground.py --mode repl --recording temp/session.json ...
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env（在其他 pce 导入之前，确保环境变量就绪）
_env_path = _project_root / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from pce.agent import PCEAgent  # noqa: E402
from pce.insight_cache import InsightCache  # noqa: E402
from pce.mock_tool_provider import MockToolProvider, RecordingProxy  # noqa: E402
from pce.models import QueryResponse  # noqa: E402

logger = logging.getLogger("playground")

# ── 兜底标记列表（与 agent.py 中 _run_react_loop 的终止路径对应）──────────
_FALLBACK_MARKERS = {
    "__REACT_TIMEOUT_BUDGET__": "超时预算耗尽",
    "__REACT_NO_TOOL_EXHAUSTED__": "无工具纠正次数耗尽",
    "__REACT_LENGTH_EXHAUSTED__": "输出截断续写次数耗尽",
    "__REACT_TIMEOUT__": "单步超时重试耗尽",
    "__REACT_DELIVER_EMPTY__": "deliver 参数为空或解析失败",
}


def _detect_status(response: QueryResponse) -> str:
    """从 QueryResponse 判断终止状态。"""
    answer = response.answer
    for marker, desc in _FALLBACK_MARKERS.items():
        if marker in answer:
            return f"异常终止: {desc}"
    return "deliver (正常终止)"


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... (共 {len(text)} 字符)"


def _format_tool_stats(stats: dict[str, Any]) -> str:
    """格式化工具调用统计。"""
    parts = []
    per_tool = stats.get("per_tool", {})
    for name, count in sorted(per_tool.items(), key=lambda x: -x[1]):
        parts.append(f"{name} x{count}")
    return ", ".join(parts) if parts else "无"


def _print_result(
    response: QueryResponse,
    elapsed_s: float,
    stats: dict[str, Any] | None = None,
) -> None:
    """格式化输出查询结果和观测信息。"""
    status = _detect_status(response)
    total_calls = stats.get("total", 0) if stats else "N/A"

    print()
    print("═" * 60)
    print(f"  状态: {status}")
    print(f"  会话: {response.session_id}")
    print(f"  耗时: {elapsed_s:.1f}s")
    print(f"  工具调用: {total_calls}")
    if stats:
        print(f"  调用明细: {_format_tool_stats(stats)}")
        if stats.get("errors", 0) > 0:
            print(f"  工具错误: {stats['errors']}")
        if stats.get("replay_hits", 0) > 0:
            print(f"  预录命中: {stats['replay_hits']}")
        if stats.get("misses", 0) > 0:
            print(f"  未命中: {stats['misses']}")
    print("─" * 60)
    print(f"  回答:\n{_truncate(response.answer)}")
    print("═" * 60)
    print()


# ============================================================================
# 核心查询函数
# ============================================================================


async def run_single_query(
    question: str,
    tool_provider: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    max_seconds: float = 300,
    memory_root: Path | None = None,
    insight_cache: InsightCache | None = None,
) -> tuple[QueryResponse, float]:
    """执行单次查询，新建 agent 实例（验证无状态设计）。"""
    agent = PCEAgent(
        model=model,
        provider=provider,
        max_seconds=max_seconds,
        insight_cache=insight_cache,
    )

    start = time.monotonic()
    response = await agent.query(
        question=question,
        memory_root=memory_root,
        serena_client=tool_provider,  # type: ignore[arg-type]  # duck typing
    )
    elapsed = time.monotonic() - start

    return response, elapsed


# ============================================================================
# Mock 模式
# ============================================================================


async def run_mock(args: argparse.Namespace) -> None:
    """Mock 模式：使用 MockToolProvider 执行单次查询。"""
    recording_path = Path(args.recording) if args.recording else None
    project_path = Path(args.memory_root) if args.memory_root else _project_root
    provider = MockToolProvider(
        project_path=str(project_path),
        recording_path=recording_path,
        strict=args.strict,
    )

    # 若 --project-path 指定了真实项目目录，启用 InsightCache
    insight_cache: InsightCache | None = None
    cache_root = Path(args.project_path) if getattr(args, "project_path", None) else None
    if cache_root and cache_root.is_dir():
        insight_cache = InsightCache(project_root=cache_root)
        await insight_cache.ensure_layout()
        logger.info("InsightCache 已启用: %s", cache_root / ".pce" / "insights")

    logger.info(
        "Mock 模式启动 (规则: %d, 预录: %d)",
        len(provider._rules),
        len(provider._replay_queue),
    )

    response, elapsed = await run_single_query(
        question=args.query,
        tool_provider=provider,
        model=args.model,
        provider=args.provider,
        max_seconds=args.max_seconds,
        memory_root=project_path,
        insight_cache=insight_cache,
    )

    _print_result(response, elapsed, provider.stats)

    if insight_cache is not None:
        stats = await insight_cache.stats()
        print(
            f"\n[InsightCache] 总条目: {stats.total_entries}  活跃: {stats.active_entries}  过时: {stats.stale_entries}"
        )


# ============================================================================
# REPL 模式
# ============================================================================


async def run_repl(args: argparse.Namespace) -> None:
    """REPL 模式：连接真实 Serena，交互式测试。"""
    from pce.serena_client import SerenaClient

    project_path = Path(args.project_path).resolve()
    serena_path = Path(args.serena_path).resolve()

    print(f"正在连接 Serena (project={project_path}, serena={serena_path})...")
    client = SerenaClient()
    try:
        await client.connect(project_path, serena_path)
    except Exception as exc:
        print(f"Serena 连接失败: {exc}")
        return

    print(f"Serena 已连接，工具数: {len(client.tools_schema)}")

    # 可选：用 RecordingProxy 包装
    provider: Any = client
    recorder: RecordingProxy | None = None
    if args.recording:
        recorder = RecordingProxy(client)
        provider = recorder
        print(f"录制模式已启用，完成后用 :save <path> 或退出时自动保存到 {args.recording}")

    # 启用 InsightCache
    insight_cache = InsightCache(project_root=project_path)
    await insight_cache.ensure_layout()
    print(f"InsightCache 已启用: {project_path / '.pce' / 'insights'}")

    # 可选的 prompt-template
    if args.prompt_template:
        _apply_prompt_template(Path(args.prompt_template))

    # 如果提供了 --query，先执行一次
    if args.query:
        await _repl_run_query(args, provider, recorder, insight_cache=insight_cache)

    # REPL 循环
    print("\n输入查询或特殊命令 (:reload, :save <path>, :stats, :quit)")
    while True:
        try:
            user_input = input("\nPCE> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # 特殊命令
        if user_input == ":quit":
            break
        if user_input == ":reload":
            if args.prompt_template:
                _apply_prompt_template(Path(args.prompt_template))
                print("System prompt 模板已重新加载")
            else:
                print("未指定 --prompt-template，无法重新加载")
            continue
        if user_input == ":stats":
            if recorder:
                print(f"录制统计: {recorder.stats}")
                print(f"录制条数: {len(recorder.call_log)}")
            else:
                print("未启用录制模式")
            continue
        if user_input.startswith(":save"):
            if recorder is None:
                print("未启用录制模式")
                continue
            parts = user_input.split(maxsplit=1)
            save_path = parts[1] if len(parts) > 1 else args.recording
            if not save_path:
                print("用法: :save <path>")
                continue
            recorder.save(save_path)
            print(f"录制已保存: {save_path}")
            continue

        # 普通查询
        try:
            response, elapsed = await run_single_query(
                question=user_input,
                tool_provider=provider,
                model=args.model,
                provider=args.provider,
                max_seconds=args.max_seconds,
                memory_root=project_path,
                insight_cache=insight_cache,
            )
            stats = recorder.stats if recorder else None
            _print_result(response, elapsed, stats)
        except Exception as exc:
            logger.exception("查询执行失败")
            print(f"错误: {exc}")

    # 退出时自动保存录制
    if recorder and args.recording:
        recorder.save(args.recording)
        print(f"录制已自动保存: {args.recording}")

    await client.disconnect()
    print("Serena 已断开连接")


async def _repl_run_query(
    args: argparse.Namespace,
    provider: Any,
    recorder: RecordingProxy | None,
    *,
    insight_cache: InsightCache | None = None,
) -> None:
    """在 REPL 模式下执行 --query 提供的首条查询。"""
    try:
        response, elapsed = await run_single_query(
            question=args.query,
            tool_provider=provider,
            model=args.model,
            provider=args.provider,
            max_seconds=args.max_seconds,
            memory_root=(
                Path(args.project_path).resolve() if getattr(args, "project_path", None) else None
            ),
            insight_cache=insight_cache,
        )
        stats = recorder.stats if recorder else None
        _print_result(response, elapsed, stats)
    except Exception as exc:
        logger.exception("首条查询执行失败")
        print(f"错误: {exc}")


# ============================================================================
# Prompt 模板切换
# ============================================================================


def _apply_prompt_template(template_path: Path) -> None:
    """读取模板文件内容，替换 agent 模块的 SYSTEM_PROMPT_HEADER 常量。"""
    import pce.agent as agent_mod

    if not template_path.exists():
        print(f"模板文件不存在: {template_path}")
        return

    content = template_path.read_text(encoding="utf-8")
    agent_mod.SYSTEM_PROMPT_HEADER = content  # type: ignore[attr-defined]
    print(f"System prompt 已替换为: {template_path} ({len(content)} 字符)")


# ============================================================================
# CLI 参数解析
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PCE Agent 独立测试环境",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Mock 模式
  uv run python scripts/agent_playground.py --mode mock --query "查找 PCEAgent 的定义"

  # REPL 模式
  uv run python scripts/agent_playground.py --mode repl --project-path . --serena-path ../serena

  # REPL + 录制
  uv run python scripts/agent_playground.py --mode repl --recording temp/rec.json ...
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "repl"],
        required=True,
        help="运行模式: mock（离线）或 repl（真实 Serena）",
    )
    parser.add_argument(
        "--query",
        default="",
        help="查询内容（mock 模式必选，repl 模式可选作首条查询）",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="覆盖 PCE_PROVIDER 环境变量（如 openrouter / openai / anthropic）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖 PCE_MODEL 环境变量（如 openai/gpt-5 或 gpt-4o-mini）",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300,
        help="推理时间上限（秒，默认 300）",
    )
    parser.add_argument(
        "--memory-root",
        default="",
        help=".pce 目录的父路径（默认：项目根目录）",
    )
    parser.add_argument(
        "--prompt-template",
        default="",
        help="自定义 system prompt 模板文件路径",
    )
    parser.add_argument(
        "--recording",
        default="",
        help="录制文件路径（mock: 加载回放; repl: 录制到文件）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="mock 模式: 工具未命中时报错（默认返回空结果）",
    )
    parser.add_argument(
        "--project-path",
        default=".",
        help="repl 模式: 目标项目路径",
    )
    parser.add_argument(
        "--serena-path",
        default=os.getenv("SERENA_PATH", str(_project_root / "serena")),
        help="repl 模式: Serena 安装路径",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )

    return parser


# ============================================================================
# 主入口
# ============================================================================


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 可选的 prompt-template
    if args.prompt_template:
        _apply_prompt_template(Path(args.prompt_template))

    if args.mode == "mock":
        if not args.query:
            parser.error("mock 模式需要 --query 参数")
        await run_mock(args)
    elif args.mode == "repl":
        await run_repl(args)


if __name__ == "__main__":
    asyncio.run(main())
