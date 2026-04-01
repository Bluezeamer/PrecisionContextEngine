from __future__ import annotations

import hashlib
import re
from pathlib import Path

_COMMENT_PATTERNS = (
    re.compile(r"^\s*#"),
    re.compile(r"^\s*//"),
    re.compile(r"^\s*/\*"),
    re.compile(r"^\s*\*"),
    re.compile(r"^\s*\*/"),
)


def normalize_source_text(text: str) -> str:
    """生成用于实质变更判断的轻量归一化文本。

    只做确定性降噪：
    - 去空行
    - 去行首尾空白
    - 去常见整行注释
    """
    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.match(line) for pattern in _COMMENT_PATTERNS):
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def fingerprint_text(text: str) -> str:
    normalized = normalize_source_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        content = path.read_text("utf-8", errors="ignore")
    except OSError:
        return None
    return fingerprint_text(content)
