from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pce.file_discovery import (
    filter_trackable_files,
    should_track_deleted_path,
    should_track_existing_file,
    supports_symbol_index,
)


def _write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, "utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "ignored-by-project/\n*.secret\n")
        _write(root / ".pce" / "pceignore", "ignored-by-pce/\n*.cache.txt\n")

        _write(root / "src" / "app.vue", "<template>Hello</template>\n")
        _write(root / "notes.md", "# note\n")
        _write(root / "ignored-by-project" / "a.py", "print('x')\n")
        _write(root / "ignored-by-pce" / "b.py", "print('y')\n")
        _write(root / "config.secret", "token=abc\n")
        _write(root / "blob.bin", b"\x00\x01\x02\x03")

        discovered = filter_trackable_files(
            root,
            [
                "src/app.vue",
                "notes.md",
                "ignored-by-project/a.py",
                "ignored-by-pce/b.py",
                "config.secret",
                "blob.bin",
            ],
        )

        assert "src/app.vue" in discovered
        assert "notes.md" in discovered
        assert "ignored-by-project/a.py" not in discovered
        assert "ignored-by-pce/b.py" not in discovered
        assert "config.secret" not in discovered
        assert "blob.bin" not in discovered

        assert should_track_existing_file(root, "src/app.vue") is True
        assert should_track_existing_file(root, "blob.bin") is False
        assert should_track_deleted_path("ignored-by-project/a.py") is True
        assert supports_symbol_index("src/app.vue") is False
        assert supports_symbol_index("server.py") is True

    print(json.dumps({
        "ok": True,
        "tests": [
            "project .gitignore 与 .pce/pceignore 均生效",
            "有效文本文件不再依赖扩展名白名单",
            "二进制文件会被过滤",
            "符号索引与文件发现已解耦",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
