"""Serena 语言健康检查与 init 前自愈。"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from .file_discovery import (
    is_hard_skipped,
    is_ignored_by_project_gitignore,
    is_probably_text_file,
)
from .models import LanguageHealthReport, LanguageSupportIssue
from .serena_client import SerenaClient, SerenaClientError
from .serena_language_registry import (
    SNAPSHOT_VERSION,
    detect_languages_from_paths,
    get_language_spec,
    sort_language_keys,
)

_PROJECT_YML = Path(".serena/project.yml")
_PROJECT_LOCAL_YML = Path(".serena/project.local.yml")
_LANGUAGES_KEY = "languages"
_WARN_FALLBACK_MESSAGE = (
    "对应语言的符号索引不可用，query / impact 将退化为文本检索，耗时与分析质量都会下降。"
)


def _discover_candidate_files(project_root: Path) -> list[str]:
    """在 baseline 建立前做本地文件扫描，只依赖 .gitignore 与硬跳过规则。"""
    results: list[str] = []
    for current_root, dirs, files in os.walk(project_root):
        rel_root = Path(current_root).resolve().relative_to(project_root).as_posix()
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = [
            name
            for name in dirs
            if not is_hard_skipped(Path(rel_root) / name)
            and not is_ignored_by_project_gitignore(project_root, Path(rel_root) / name)
        ]
        for name in files:
            rel_path = (Path(rel_root) / name).as_posix().lstrip("./")
            if is_hard_skipped(rel_path):
                continue
            if is_ignored_by_project_gitignore(project_root, rel_path):
                continue
            abs_path = project_root / rel_path
            if not abs_path.is_file():
                continue
            if not is_probably_text_file(abs_path):
                continue
            results.append(rel_path)
    return sorted(results)


def _extract_languages_from_yaml_text(text: str) -> list[str] | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != f"{_LANGUAGES_KEY}:":
            continue
        values: list[str] = []
        cursor = idx + 1
        while cursor < len(lines):
            raw = lines[cursor]
            stripped = raw.strip()
            if not stripped:
                break
            if stripped.startswith("#"):
                cursor += 1
                continue
            if re.match(r"^- [A-Za-z0-9_]+$", stripped):
                values.append(stripped[2:].strip())
                cursor += 1
                continue
            break
        return values
    return None


def _replace_languages_block(text: str, languages: list[str]) -> str:
    lines = text.splitlines()
    replacement = [f"{_LANGUAGES_KEY}:", *[f"- {language}" for language in languages]]

    for idx, line in enumerate(lines):
        if line.strip() != f"{_LANGUAGES_KEY}:":
            continue
        end = idx + 1
        while end < len(lines):
            stripped = lines[end].strip()
            if re.match(r"^- [A-Za-z0-9_]+$", stripped):
                end += 1
                continue
            if not stripped:
                break
            if stripped.startswith("#"):
                end += 1
                continue
            break
        updated = [*lines[:idx], *replacement, *lines[end:]]
        return "\n".join(updated).rstrip() + "\n"

    insert_at = 1 if lines and lines[0].startswith("project_name:") else 0
    updated = [*lines[:insert_at], *replacement, *lines[insert_at:]]
    return "\n".join(updated).rstrip() + "\n"


def _minimal_project_yml(project_root: Path, languages: list[str]) -> str:
    lines = [
        f'project_name: "{project_root.name}"',
        "",
        "languages:",
        *[f"- {language}" for language in languages],
        "",
        'encoding: "utf-8"',
        "ignore_all_files_in_gitignore: true",
        "ls_specific_settings: {}",
        "ignored_paths: []",
        "read_only: false",
        "",
    ]
    return "\n".join(lines)


async def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, content, "utf-8")


def _build_fix_suggestions(language: str, missing_commands: list[str], reason: str) -> list[str]:
    spec = get_language_spec(language)
    if spec is None:
        return [reason, "检查 Serena project.yml 中的语言配置后重新执行 pce_init。"]
    if spec.install_channel == "npm":
        runtime = "、".join(missing_commands or spec.required_commands or ("node", "npm"))
        return [
            f"安装 {runtime}，并确认相关命令在 PATH 中可执行。",
            f"重新执行 pce_init，Serena 将自动安装 {language} 对应的 npm 运行时依赖。",
        ]
    if spec.install_channel == "system_toolchain":
        runtime = "、".join(missing_commands or spec.required_commands or ("对应工具链",))
        return [
            f"安装或修复 {runtime} 所在的系统工具链。",
            "确认对应命令可执行后重新执行 pce_init。",
        ]
    return [
        "检查 Serena 语言运行时与系统依赖是否已安装。",
        "必要时手动编辑 .serena/project.yml / project.local.yml 后重新执行 pce_init。",
    ]


def _build_warning_text(issue: LanguageSupportIssue) -> str:
    reason = issue.reason
    suggestions = "；".join(issue.suggested_fix[:2]) if issue.suggested_fix else "请检查语言运行时与 Serena 配置。"
    return (
        f"{issue.language} 语言支持未达成：{reason}。"
        f"{' '.join(issue.impact) if issue.impact else _WARN_FALLBACK_MESSAGE} "
        f"修复建议：{suggestions}"
    )


def _make_issue(
    *,
    language: str,
    status: str,
    failure_stage: str,
    reason_code: str,
    reason: str,
    representative_files: list[str],
    auto_fix_attempted: bool,
    auto_fix_result: str,
    suggested_fix: list[str],
) -> LanguageSupportIssue:
    spec = get_language_spec(language)
    return LanguageSupportIssue(
        language=language,
        status=status,
        failure_stage=failure_stage,
        reason_code=reason_code,
        reason=reason,
        auto_fix_attempted=auto_fix_attempted,
        auto_fix_result=auto_fix_result,
        required_runtime=list(spec.required_commands) if spec else [],
        install_channel=spec.install_channel if spec else None,
        representative_files=representative_files,
        suggested_fix=suggested_fix,
        impact=[_WARN_FALLBACK_MESSAGE],
    )


async def preflight_serena_language_health(project_root: Path) -> LanguageHealthReport:
    """在 Serena 启动前校验并修复项目语言配置。"""
    files = _discover_candidate_files(project_root)
    detected = detect_languages_from_paths(files)
    project_yml_path = project_root / _PROJECT_YML
    project_local_yml_path = project_root / _PROJECT_LOCAL_YML

    project_text = project_yml_path.read_text("utf-8") if project_yml_path.exists() else ""
    local_text = project_local_yml_path.read_text("utf-8") if project_local_yml_path.exists() else ""
    project_languages = _extract_languages_from_yaml_text(project_text) or []
    local_languages = _extract_languages_from_yaml_text(local_text)

    effective_before = sort_language_keys(local_languages if local_languages is not None else project_languages)
    config_target_path = (
        project_local_yml_path if local_languages is not None else project_yml_path
    )

    issues: list[LanguageSupportIssue] = []
    configured_after = list(effective_before)
    repaired_languages: list[str] = []

    for language in detected.detected_languages:
        if language in configured_after:
            continue
        spec = get_language_spec(language)
        representative_files = list(detected.representative_files.get(language, ()))
        if spec is None:
            issues.append(
                _make_issue(
                    language=language,
                    status="manual_action_required",
                    failure_stage="language_detection",
                    reason_code="unknown_snapshot_language",
                    reason="语言存在于项目文件中，但不在当前 PCE Serena 语言快照中。",
                    representative_files=representative_files,
                    auto_fix_attempted=False,
                    auto_fix_result="skipped",
                    suggested_fix=[
                        "更新 PCE 内置的 Serena 语言快照表后重新执行 pce_init。",
                    ],
                )
            )
            continue
        if not spec.auto_enable:
            issues.append(
                _make_issue(
                    language=language,
                    status="manual_action_required",
                    failure_stage="runtime_precheck",
                    reason_code="auto_enable_not_supported",
                    reason="当前 PCE 语言快照未为该语言提供自动启用策略。",
                    representative_files=representative_files,
                    auto_fix_attempted=False,
                    auto_fix_result="skipped",
                    suggested_fix=_build_fix_suggestions(
                        language,
                        [],
                        "需要手动补齐对应语言运行时和 Serena 配置。",
                    ),
                )
            )
            continue

        missing_commands = [cmd for cmd in spec.required_commands if shutil.which(cmd) is None]
        if missing_commands:
            issues.append(
                _make_issue(
                    language=language,
                    status="degraded",
                    failure_stage="runtime_precheck",
                    reason_code="missing_system_dependency",
                    reason=f"缺少运行时命令: {', '.join(missing_commands)}。",
                    representative_files=representative_files,
                    auto_fix_attempted=True,
                    auto_fix_result="skipped",
                    suggested_fix=_build_fix_suggestions(
                        language,
                        missing_commands,
                        "缺少语言运行时。",
                    ),
                )
            )
            continue

        configured_after.append(language)
        repaired_languages.append(language)

    configured_after = sort_language_keys(configured_after)

    if configured_after != effective_before:
        try:
            if config_target_path.exists():
                current_text = config_target_path.read_text("utf-8")
                updated_text = _replace_languages_block(current_text, configured_after)
            else:
                updated_text = _minimal_project_yml(project_root, configured_after)
            await _write_text(config_target_path, updated_text)
        except Exception as exc:
            failed_languages = [
                language for language in repaired_languages if language not in effective_before
            ]
            configured_after = list(effective_before)
            for language in failed_languages:
                representative_files = list(detected.representative_files.get(language, ()))
                issues.append(
                    _make_issue(
                        language=language,
                        status="degraded",
                        failure_stage="config_repair",
                        reason_code="config_write_failed",
                        reason=f"写入 {config_target_path} 失败: {exc}",
                        representative_files=representative_files,
                        auto_fix_attempted=True,
                        auto_fix_result="failed",
                        suggested_fix=[
                            f"检查 {config_target_path} 的写权限后重新执行 pce_init。",
                            "必要时手动补齐 Serena languages 配置。",
                        ],
                    )
                )
            repaired_languages = []

    effective_config_path = str(config_target_path if config_target_path.exists() else project_yml_path)
    warnings = [_build_warning_text(item) for item in issues if item.status != "repaired"]

    return LanguageHealthReport(
        snapshot_version=SNAPSHOT_VERSION,
        detected_languages=list(detected.detected_languages),
        configured_languages_before=list(effective_before),
        configured_languages_after=list(configured_after),
        repaired_languages=repaired_languages,
        project_config_path=str(project_yml_path),
        project_local_override_path=str(project_local_yml_path) if project_local_yml_path.exists() else None,
        effective_config_path=effective_config_path,
        representative_files={
            key: list(value) for key, value in detected.representative_files.items()
        },
        issues=issues,
        warnings=warnings,
    )


def _extract_symbol_error(payload: object) -> str | None:
    if isinstance(payload, str) and payload.startswith("Error executing tool:"):
        return payload
    if isinstance(payload, dict):
        structured = payload.get("structuredContent")
        if isinstance(structured, dict):
            result = structured.get("result")
            error = _extract_symbol_error(result)
            if error:
                return error
        content = payload.get("content")
        if isinstance(content, list):
            for item in content:
                error = _extract_symbol_error(item)
                if error:
                    return error
        text = payload.get("text")
        if isinstance(text, str) and text.startswith("Error executing tool:"):
            return text
    if isinstance(payload, list):
        for item in payload:
            error = _extract_symbol_error(item)
            if error:
                return error
    return None


async def verify_serena_language_health(
    report: LanguageHealthReport,
    serena_client: SerenaClient,
) -> LanguageHealthReport:
    """Serena 启动后做轻量抽样校验。"""
    issues = list(report.issues)
    warnings = list(report.warnings)
    degraded_languages = {item.language for item in issues if item.status != "repaired"}

    for language in report.configured_languages_after:
        if language in degraded_languages:
            continue
        samples = report.representative_files.get(language, [])
        if not samples:
            continue
        try:
            raw = await serena_client.get_symbols_overview(samples[0], depth=1)
            error = _extract_symbol_error(raw)
        except SerenaClientError as exc:
            error = str(exc)

        if error:
            issue = _make_issue(
                language=language,
                status="degraded",
                failure_stage="post_start_verification",
                reason_code="symbol_probe_failed",
                reason=f"代表文件 `{samples[0]}` 的符号探测失败: {error}",
                representative_files=list(samples),
                auto_fix_attempted=True,
                auto_fix_result="failed",
                suggested_fix=_build_fix_suggestions(
                    language,
                    [],
                    "Serena 启动后仍无法提取符号。",
                ),
            )
            issues.append(issue)
            warnings.append(_build_warning_text(issue))

    return report.model_copy(
        update={
            "issues": issues,
            "warnings": warnings,
        }
    )
