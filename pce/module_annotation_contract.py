"""模块认知文档的轻量契约与校验辅助。"""

from __future__ import annotations

from typing import Any

FIXED_MODULE_SECTION_HEADINGS: tuple[str, ...] = (
    "覆盖文件",
    "核心职责",
    "关键流程",
    "外部协作",
    "风险与约束",
)
OPTIONAL_MODULE_SECTION_HEADINGS: tuple[str, ...] = ("关键符号",)


def parse_module_annotation_markdown(content: str) -> dict[str, Any]:
    """解析模块认知 Markdown，返回 title 与 sections。"""
    title = ""
    current_heading: str | None = None
    current_body: list[str] = []
    sections: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal current_heading, current_body
        if current_heading is None:
            return
        sections.append({
            "heading": current_heading,
            "body_lines": current_body[:],
        })
        current_heading = None
        current_body = []

    for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            _flush()
            current_heading = line[3:].strip()
            current_body = []
            continue
        if current_heading is not None:
            current_body.append(line)
    _flush()

    return {
        "title": title,
        "sections": sections,
        "sections_map": {section["heading"]: section["body_lines"] for section in sections},
    }


def extract_coverage_file_paths(content: str) -> list[str]:
    parsed = parse_module_annotation_markdown(content)
    lines = parsed["sections_map"].get("覆盖文件", [])
    paths: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            path = stripped[2:].strip()
            if path:
                paths.append(path)
    return paths


def validate_module_annotation_markdown(
    content: str,
    *,
    expected_file_paths: list[str],
    require_core_responsibility: bool = True,
) -> list[str]:
    """校验模块认知 Markdown 的基本结构与覆盖文件。"""
    errors: list[str] = []
    parsed = parse_module_annotation_markdown(content)

    if not parsed["title"]:
        errors.append("缺少一级标题 `# 模块名`")

    headings = {section["heading"] for section in parsed["sections"]}
    missing = [heading for heading in FIXED_MODULE_SECTION_HEADINGS if heading not in headings]
    if missing:
        errors.append("缺少固定章节: " + ", ".join(missing))

    coverage_paths = extract_coverage_file_paths(content)
    missing_paths = [path for path in expected_file_paths if path not in coverage_paths]
    extra_paths = [path for path in coverage_paths if path not in expected_file_paths]
    if missing_paths:
        errors.append("覆盖文件缺失: " + ", ".join(missing_paths))
    if extra_paths:
        errors.append("覆盖文件包含未知路径: " + ", ".join(extra_paths))

    if require_core_responsibility:
        body = parsed["sections_map"].get("核心职责", [])
        if not any(line.strip() for line in body):
            errors.append("`## 核心职责` 不能为空")

    return errors
