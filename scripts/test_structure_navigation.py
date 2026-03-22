"""
structure.md / index.md 静态导航回归脚本。

目标：
1. 验证 structure.md 已切换为结构导航视图，而不是顶层模块枚举表；
2. 验证 index.md 渲染会收敛多余空行；
3. 验证静态注入块在 prompt 中会做格式级紧凑化。

运行：
    uv run python scripts/test_structure_navigation.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from pce.agent import PCEAgent
from pce.indexer import (
    _build_structure_refresh_signals,
    _build_structure_rule_bundle,
    _is_cautious_structure_markdown,
    _render_index_md,
    _render_rule_structure_md,
    _should_refresh_structure,
)
from pce.models import FileMeta, IndexEntry, ModuleRegistry


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _entry(path: str, language: str, loc: int, symbols: int = 0) -> IndexEntry:
    return IndexEntry(
        file_meta=FileMeta(
            path=Path(path),
            language=language,
            size_bytes=128,
            mtime=datetime.now(UTC),
            loc=loc,
        ),
        symbols=[],
        imports=[],
        edges=[],
    )


async def _test_structure_navigation_layout() -> None:
    entries = [
        _entry("main.py", "python", 20),
        _entry("backend/app.py", "python", 120),
        _entry("frontend/src/main.js", "javascript", 40),
        _entry("frontend/src/App.vue", "javascript", 60),
        _entry("PrecisionContextEngine/pce/agent.py", "python", 300),
        _entry("PrecisionContextEngine/pce/server.py", "python", 240),
        _entry("PrecisionContextEngine/scripts/test_e2e.py", "python", 80),
    ]
    bundle = _build_structure_rule_bundle(entries, index_sections=[])
    content = _render_rule_structure_md(bundle)
    _assert("# 项目结构导航" in content, "structure.md 标题未切换为结构导航")
    _assert("## 项目形态候选" in content, "规则 structure 缺少项目形态候选")
    _assert("## 顶层区域候选" in content, "规则 structure 缺少顶层区域候选")
    _assert("## 入口点候选" in content, "规则 structure 缺少入口点候选")
    _assert("## 高密度代码区域候选" in content, "规则 structure 缺少高密度代码区域候选")
    _assert("## 模块对齐提示" in content, "规则 structure 缺少模块对齐提示")
    _assert("## 导航建议" in content, "structure.md 缺少导航建议")
    _assert("顶层模块清单" not in content, "structure.md 不应再输出顶层模块清单")
    _assert("`main.py`" in content, "structure.md 应识别根级入口")
    _assert("文件名命中 `main.*` 规则" in content, "structure.md 应保留入口识别依据")
    _assert("`frontend/`" in content, "structure.md 应覆盖顶层区域候选")
    _assert("候选：" in content, "规则 structure 应使用候选式表达")


async def _test_index_render_compaction() -> None:
    rendered = _render_index_md(
        "# 项目认知导航",
        [
            {
                "name": "Agent Core",
                "body_lines": [
                    "",
                    "文件：pce/agent.py",
                    "",
                    "",
                    "职责：负责主循环。",
                    "",
                    "详细认知：.pce/annotations/modules/agent-core.md",
                    "",
                ],
            }
        ],
    )
    _assert("\n\n\n" not in rendered, "index.md 渲染后不应保留连续多余空行")


async def _test_prompt_static_block_compaction() -> None:
    raw = "\n\n# 标题\n\n\n段落A\n\n\n\n段落B\n\n"
    compacted = PCEAgent._compact_markdown_block(raw)
    _assert(compacted.startswith("# 标题"), "紧凑化后应保留正文内容")
    _assert("\n\n\n" not in compacted, "紧凑化后不应保留三连空行")


async def _test_structure_refresh_signals() -> None:
    entries = [
        _entry("main.py", "python", 20),
        _entry("backend/app.py", "python", 120),
        _entry("frontend/src/main.js", "javascript", 40),
    ]
    bundle = _build_structure_rule_bundle(
        entries,
        index_sections=[
            {
                "name": "Backend API",
                "slug": "backend-api",
                "file_paths": ["backend/app.py"],
                "body_lines": [],
            }
        ],
    )
    signals = _build_structure_refresh_signals(
        bundle,
        index_sections=[
            {
                "name": "Backend API",
                "slug": "backend-api",
                "file_paths": ["backend/app.py"],
                "body_lines": [],
            }
        ],
        registry=ModuleRegistry(),
    )
    reasons = _should_refresh_structure(
        existing_state=None,
        current_signals=signals,
        structure_exists=False,
        force=False,
    )
    _assert("structure_missing" in reasons, "structure 缺失时应触发重算")
    _assert("state_missing" in reasons, "状态缺失时应触发重算")


async def _test_structure_cautious_guard() -> None:
    accepted = """# 项目结构导航

## 项目形态概览
- 可初步视为多区域协作型项目；线索：顶层目录分布较散。

## 顶层区域
- `src/`：可优先查看；提示：文件数较高。

## 关键入口候选
- `main.rs`：入口候选；线索：文件名命中 `main.*`。

## 模块对齐提示
- `src/`：可视为主模块候选聚集区域。

## 导航建议
- 建议先查看 `src/`，若线索不足再回到模块导航。
"""
    rejected = """# 项目结构导航

## 项目形态概览
这是一个明确的前后端分层项目，由三层核心系统构成。

## 顶层区域
- `src/` 负责全部核心实现。

## 关键入口候选
- `main.rs`

## 模块对齐提示
- `src/` 对应主模块。

## 导航建议
- 先看 `src/`。
"""
    rejected_mixed = """# 项目结构导航

## 项目形态概览
- 可视为多区域协作型项目；线索：顶层目录分布较散。
- 核心逻辑集中在 Python 模块。

## 顶层区域
- `src/`：可优先查看；提示：文件数较高。

## 关键入口候选
- `main.rs`：入口候选；线索：文件名命中 `main.*`。

## 模块对齐提示
- `src/`：可视为主模块候选聚集区域。

## 导航建议
- 建议先查看 `src/`，若线索不足再回到模块导航。
"""
    _assert(_is_cautious_structure_markdown(accepted), "弱提示 style 的 structure 应通过校验")
    _assert(not _is_cautious_structure_markdown(rejected), "强断言 style 的 structure 应被拒绝")
    _assert(
        not _is_cautious_structure_markdown(rejected_mixed),
        "即使整体通过，单条 bullet 过于断言式时也应被拒绝",
    )


async def main() -> None:
    await _test_structure_navigation_layout()
    await _test_index_render_compaction()
    await _test_prompt_static_block_compaction()
    await _test_structure_refresh_signals()
    await _test_structure_cautious_guard()
    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "rule structure 已切换为弱提示候选骨架",
                    "index.md 渲染会收敛多余空行",
                    "静态注入块会做格式级紧凑化",
                    "structure 重算判定会在缺文件/缺状态时触发",
                    "LLM structure 输出必须满足弱提示式谨慎表达",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
