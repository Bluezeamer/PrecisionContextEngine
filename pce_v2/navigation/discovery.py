from __future__ import annotations

import os
from pathlib import Path

import pathspec

HARD_SKIP_DIRS = frozenset(
    {
        ".git",
        ".pce",
        ".serena",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
        ".css",
        ".scss",
        ".html",
        ".sql",
        ".sh",
        ".env",
    }
)


class DiscoveryPolicy:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.root_ignore = self._load_spec(self.project_root / ".gitignore")
        self.pce_ignore = self._load_spec(self.project_root / ".pce" / "pceignore")

    def is_visible(self, rel_path: str) -> bool:
        path = Path(rel_path)
        if any(part in HARD_SKIP_DIRS for part in path.parts):
            return False
        if self.root_ignore and self.root_ignore.match_file(rel_path):
            return False
        if self.pce_ignore and self.pce_ignore.match_file(rel_path):
            return False
        return True

    @staticmethod
    def _load_spec(path: Path) -> pathspec.PathSpec | None:
        if not path.exists():
            return None
        lines = [
            line.strip()
            for line in path.read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            return None
        return pathspec.GitIgnoreSpec.from_lines(lines)


def is_probably_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name.startswith(".") and not path.suffix:
        return True
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return False
    return b"\x00" not in sample


def discover_trackable_files(project_root: Path) -> list[str]:
    root = project_root.resolve()
    policy = DiscoveryPolicy(root)
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in HARD_SKIP_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            file_path = base / filename
            rel_path = file_path.relative_to(root).as_posix()
            if not policy.is_visible(rel_path):
                continue
            if not is_probably_text_file(file_path):
                continue
            results.append(rel_path)
    return sorted(set(results))
