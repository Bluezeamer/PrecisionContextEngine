from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pce.indexer import (
    _build_fallback_pceignore_patterns,
    _build_pceignore_prompt,
    _collect_pceignore_candidates,
    _filter_safe_pceignore_patterns,
    _parse_pceignore_patterns,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".gitignore").write_text("node_modules/\n.DS_Store\n", "utf-8")
        prompt = _build_pceignore_prompt(
            root,
            {
                "dirs": [
                    "frontend",
                    "frontend/dist",
                    "temp",
                    "docs",
                ],
                "files": [
                    "frontend/dist/bundle.js",
                    "temp/mockup.png",
                    "docs/design.md",
                    "frontend/src/App.vue",
                ],
            },
        )
        assert "项目根 .gitignore 摘要" in prompt
        assert "目录树摘要" in prompt
        assert "非常保守" in prompt
        assert "不要忽略显然属于源码" in prompt
        assert "允许选择的候选规则" in prompt

    parsed = _parse_pceignore_patterns(
        """```text
        # comment
        temp/
        *.log
        frontend/dist/
        1. docs/generated/
        ```"""
    )
    assert parsed == ["temp/", "*.log", "frontend/dist/", "docs/generated/"]

    fallback = _build_fallback_pceignore_patterns(
        {
            "dirs": ["temp", "docs", "frontend/.cache", "reports"],
            "files": ["runtime.log", "README.md", ".DS_Store"],
        }
    )
    assert fallback == ["temp/", "frontend/.cache/", "reports/", ".DS_Store", "*.log"]

    payload = {
        "dirs": ["temp", "reports", "frontend/src"],
        "files": ["runtime.log", "frontend/src/App.vue"],
    }
    candidates = _collect_pceignore_candidates(payload)
    assert candidates == ["temp/", "reports/", "*.log"]
    filtered = _filter_safe_pceignore_patterns(
        ["temp/", "*.log", "frontend/src/", "README.md"],
        payload,
    )
    assert filtered == ["temp/", "*.log"]

    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "pceignore prompt 包含保守补充黑名单约束",
                    "目录树与项目 gitignore 摘要会注入 prompt",
                    "pceignore 解析器能去掉 fence/comment/编号噪声",
                    "LLM 不可用时仍可保守推断最小黑名单",
                    "LLM 输出会被候选规则集约束过滤",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
