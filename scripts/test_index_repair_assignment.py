"""
index 二阶段补归属回归脚本。

目标：
1. 验证遗漏文件的 repair decision 解析会约束在候选路径与已有 slug 范围内；
2. 验证 attach / create 两类动作能正确更新 section；
3. 验证未命中的文件仍会留给最终 fallback。

运行：
    uv run python scripts/test_index_repair_assignment.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pce.indexer import (
    _apply_missing_coverage_repair_decisions,
    _build_missing_entry_fact,
    _dedupe_semantic_sections,
    _parse_missing_coverage_repair_decisions,
)
from pce.models import FileMeta, IndexEntry


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _entry(path: str, language: str = "python", loc: int = 20) -> IndexEntry:
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


async def _test_decision_parse_and_apply() -> None:
    content = json.dumps(
        {
            "decisions": [
                {
                    "path": "pkg/new_helper.py",
                    "kind": "implementation",
                    "action": "attach",
                    "module_slug": "core-module",
                },
                {
                    "path": "pkg/isolated.py",
                    "kind": "documentation",
                    "action": "create",
                    "module_name": "辅助隔离逻辑",
                    "responsibility": "职责：用于承接首轮模块划分遗漏的辅助逻辑文件。",
                },
                {
                    "path": "pkg/unknown.py",
                    "kind": "unknown",
                    "action": "fallback",
                },
                {
                    "path": "pkg/outside.py",
                    "kind": "implementation",
                    "action": "attach",
                    "module_slug": "missing-slug",
                },
            ]
        },
        ensure_ascii=False,
    )
    decisions = _parse_missing_coverage_repair_decisions(
        content,
        candidate_paths={"pkg/new_helper.py", "pkg/isolated.py", "pkg/unknown.py"},
        known_slugs={"core-module"},
    )
    _assert(len(decisions) == 3, "decision 解析应过滤掉非法 slug 项")

    sections = [
        {
            "name": "Core Module",
            "slug": "core-module",
            "file_paths": ["pkg/core.py"],
            "body_lines": [
                "文件：pkg/core.py",
                "职责：核心模块。",
                "详细认知：.pce/annotations/modules/core-module.md",
            ],
        }
    ]
    entries_map = {
        "pkg/core.py": _entry("pkg/core.py"),
        "pkg/new_helper.py": _entry("pkg/new_helper.py"),
        "pkg/isolated.py": _entry("pkg/isolated.py"),
        "pkg/unknown.py": _entry("pkg/unknown.py"),
    }

    updated_sections, changed_slugs, remaining = _apply_missing_coverage_repair_decisions(
        sections,
        entries_map,
        decisions,
    )
    core = next(section for section in updated_sections if section["slug"] == "core-module")
    created = next(section for section in updated_sections if section["slug"] != "core-module")

    _assert("pkg/new_helper.py" in core["file_paths"], "attach 应并入已有模块")
    _assert(created["name"] == "辅助隔离逻辑", "create 应生成新的补充模块")
    _assert(created["file_paths"] == ["pkg/isolated.py"], "create 模块应只包含对应文件")
    _assert("core-module" in changed_slugs and created["slug"] in changed_slugs, "变更 slug 集应包含 attach 与 create")
    _assert(remaining == ["pkg/unknown.py"], "未处理文件应留给最终 fallback")


async def _test_semantic_section_dedup() -> None:
    sections = [
        {
            "name": "Frontend",
            "slug": "frontend",
            "file_paths": ["frontend/src/App.vue", "frontend/src/main.js"],
            "body_lines": [
                "文件：frontend/src/App.vue, frontend/src/main.js",
                "职责：前端主界面。",
                "详细认知：.pce/annotations/modules/frontend.md",
            ],
        },
        {
            "name": "Frontend Application",
            "slug": "frontend-application",
            "file_paths": ["frontend/src/App.vue", "frontend/src/main.js", "frontend/index.html"],
            "body_lines": [
                "文件：frontend/src/App.vue, frontend/src/main.js, frontend/index.html",
                "职责：前端应用。",
                "详细认知：.pce/annotations/modules/frontend-application.md",
            ],
        },
    ]
    deduped = _dedupe_semantic_sections(sections)
    _assert(len(deduped) == 1, "高重叠前端章节应被去重收敛")
    _assert(
        set(deduped[0]["file_paths"]) == {
            "frontend/src/App.vue",
            "frontend/src/main.js",
            "frontend/index.html",
        },
        "去重后应合并文件路径",
    )


async def _test_missing_entry_fact_windows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        short_path = root / "docs" / "note.md"
        short_path.parent.mkdir(parents=True, exist_ok=True)
        short_path.write_text("# 标题\n第一行\n第二行\n", encoding="utf-8")
        short_fact = await _build_missing_entry_fact(_entry("docs/note.md", language="markdown", loc=3), root)
        _assert("full" in short_fact["content_windows"], "短文件应直接全量注入")
        _assert(short_fact.get("type_hints"), "Markdown 文件应提取标题提示")

        long_path = root / "pkg" / "long.py"
        long_path.parent.mkdir(parents=True, exist_ok=True)
        long_content = "\n".join(f"line_{i}" for i in range(1, 101))
        long_path.write_text(long_content, encoding="utf-8")
        long_fact = await _build_missing_entry_fact(_entry("pkg/long.py", loc=100), root)
        windows = long_fact["content_windows"]
        _assert({"head", "middle", "tail"} <= set(windows.keys()), "长文件应提供三段窗口")
        _assert("line_1" in windows["head"], "head 窗口应包含前段内容")
        _assert("line_50" in windows["middle"] or "line_51" in windows["middle"], "middle 窗口应包含中段内容")
        _assert("line_100" in windows["tail"], "tail 窗口应包含尾段内容")


async def main() -> None:
    await _test_decision_parse_and_apply()
    await _test_semantic_section_dedup()
    await _test_missing_entry_fact_windows()
    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "遗漏文件 repair decision 会过滤非法 slug 与越界路径",
                    "attach / create 动作可正确更新 sections",
                    "未命中的文件会保留给最终 fallback",
                    "高重叠语义章节会被合并去重",
                    "遗漏文件 facts 包会对短文件全量注入、长文件按窗口抽样",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
