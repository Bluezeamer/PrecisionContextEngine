from __future__ import annotations

import json

from pce.indexer import _fallback_display_name_from_group, _fallback_group_key_for_file


def main() -> None:
    shell_group = _fallback_group_key_for_file("hicolors_logic_v2/__init__.py")
    root_entry_group = _fallback_group_key_for_file("main.py")
    normal_group = _fallback_group_key_for_file("frontend/src/components/ImageUploader.vue")

    assert shell_group == "shell::hicolors_logic_v2"
    assert _fallback_display_name_from_group(shell_group) == "补充归类 包级壳层 hicolors_logic_v2"
    assert root_entry_group == "shell::__root__"
    assert _fallback_display_name_from_group(root_entry_group) == "补充归类 项目入口与共享模型"
    assert normal_group == "frontend/src/components"
    assert _fallback_display_name_from_group(normal_group) == "补充归类 frontend / src / components"

    print(
        json.dumps(
            {
                "ok": True,
                "tests": [
                    "壳层文件会进入 shell fallback 分组",
                    "根级入口文件会进入项目入口与共享模型分组",
                    "普通文件仍按目录 fallback 分组",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
