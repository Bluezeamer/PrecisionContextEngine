from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pce.serena_language_health as health_mod
from pce.serena_language_health import (
    preflight_serena_language_health,
    verify_serena_language_health,
)
from pce.serena_language_registry import infer_language_for_path, supports_symbol_index_for_path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, "utf-8")


class _FakeSerenaClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses

    async def get_symbols_overview(self, relative_path: str, *, depth: int = 0) -> object:
        del depth
        return self._responses[relative_path]


async def _test_preflight_repairs_project_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "")
        _write(root / "backend" / "app.py", "def app():\n    return 1\n")
        _write(root / "frontend" / "src" / "App.vue", "<template>Hello</template>\n")
        _write(root / "frontend" / "src" / "main.js", "console.log('x')\n")
        _write(
            root / ".serena" / "project.yml",
            'project_name: "demo"\n\nlanguages:\n- python\n\nencoding: "utf-8"\n',
        )

        original = health_mod.shutil.which
        health_mod.shutil.which = lambda cmd: f"/usr/bin/{cmd}"
        try:
            report = await preflight_serena_language_health(root)
        finally:
            health_mod.shutil.which = original

        updated = (root / ".serena" / "project.yml").read_text("utf-8")
        assert report.detected_languages == ["python", "typescript", "vue"]
        assert report.configured_languages_after == ["python", "typescript", "vue"]
        assert report.repaired_languages == ["typescript", "vue"]
        assert "languages:\n- python\n- typescript\n- vue\n" in updated


async def _test_missing_runtime_reports_fix_advice() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "")
        _write(root / "frontend" / "src" / "App.vue", "<template>Hello</template>\n")
        _write(root / "frontend" / "src" / "main.js", "console.log('x')\n")
        _write(
            root / ".serena" / "project.yml",
            'project_name: "demo"\n\nlanguages:\n- python\n\nencoding: "utf-8"\n',
        )

        original = health_mod.shutil.which

        def _which(cmd: str) -> str | None:
            if cmd in {"node", "npm"}:
                return None
            return f"/usr/bin/{cmd}"

        health_mod.shutil.which = _which
        try:
            report = await preflight_serena_language_health(root)
        finally:
            health_mod.shutil.which = original

        assert report.configured_languages_after == ["python"]
        assert {item.language for item in report.issues} == {"typescript", "vue"}
        assert any("缺少运行时命令" in warning for warning in report.warnings)
        assert any("重新执行 pce_init" in warning for warning in report.warnings)


async def _test_project_local_override_is_repaired_in_place() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "")
        _write(root / "frontend" / "src" / "App.vue", "<template>Hello</template>\n")
        _write(
            root / ".serena" / "project.yml",
            'project_name: "demo"\n\nlanguages:\n- python\n\nencoding: "utf-8"\n',
        )
        _write(root / ".serena" / "project.local.yml", "languages:\n- python\n")

        original = health_mod.shutil.which
        health_mod.shutil.which = lambda cmd: f"/usr/bin/{cmd}"
        try:
            report = await preflight_serena_language_health(root)
        finally:
            health_mod.shutil.which = original

        updated_local = (root / ".serena" / "project.local.yml").read_text("utf-8")
        assert report.effective_config_path.endswith("project.local.yml")
        assert "languages:\n- python\n- typescript\n- vue\n" in updated_local


async def _test_post_start_verification_surfaces_symbol_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "")
        _write(root / "frontend" / "src" / "App.vue", "<template>Hello</template>\n")
        _write(root / "frontend" / "src" / "main.js", "console.log('x')\n")
        _write(
            root / ".serena" / "project.yml",
            'project_name: "demo"\n\nlanguages:\n- python\n\nencoding: "utf-8"\n',
        )

        original = health_mod.shutil.which
        health_mod.shutil.which = lambda cmd: f"/usr/bin/{cmd}"
        try:
            report = await preflight_serena_language_health(root)
        finally:
            health_mod.shutil.which = original

        verified = await verify_serena_language_health(
            report,
            _FakeSerenaClient(
                {
                    "frontend/src/main.js": {
                        "structuredContent": {
                            "result": "Error executing tool: ValueError - probe failed"
                        }
                    },
                    "frontend/src/App.vue": {
                        "structuredContent": {
                            "result": {"Function": ["App"]}
                        }
                    },
                }
            ),
        )
        assert any(item.failure_stage == "post_start_verification" for item in verified.issues)
        assert any("probe failed" in warning for warning in verified.warnings)


async def _test_rust_repo_is_auto_enabled_when_runtime_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root / ".gitignore", "")
        _write(root / "src" / "lib.rs", "pub fn demo() {}\n")
        _write(
            root / ".serena" / "project.yml",
            'project_name: "demo"\n\nlanguages:\n- python\n\nencoding: "utf-8"\n',
        )

        original = health_mod.shutil.which
        health_mod.shutil.which = lambda cmd: f"/usr/bin/{cmd}"
        try:
            report = await preflight_serena_language_health(root)
        finally:
            health_mod.shutil.which = original

        assert report.detected_languages == ["rust"]
        assert report.configured_languages_after == ["python", "rust"]
        assert report.repaired_languages == ["rust"]
        assert report.issues == []


def _test_registry_genericity() -> None:
    assert infer_language_for_path("backend/main.py") == "python"
    assert infer_language_for_path("frontend/src/App.vue") == "vue"
    assert infer_language_for_path("frontend/src/main.js") == "typescript"
    assert infer_language_for_path("src/lib.rs") == "rust"
    assert supports_symbol_index_for_path("src/lib.rs") is True
    assert supports_symbol_index_for_path("docs/readme.md") is False


async def main() -> None:
    await _test_preflight_repairs_project_config()
    await _test_missing_runtime_reports_fix_advice()
    await _test_project_local_override_is_repaired_in_place()
    await _test_post_start_verification_surfaces_symbol_failure()
    await _test_rust_repo_is_auto_enabled_when_runtime_available()
    _test_registry_genericity()
    print(json.dumps({"ok": True, "tests": 6}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
