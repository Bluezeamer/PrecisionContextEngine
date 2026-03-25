"""Serena 语言快照表。

目标：
1. 将 PCE 的文件语言归一化、symbol index 判定、init 前语言探测统一到一份稳定快照；
2. 避免在多个模块中手写扩展名白名单，和 Serena 自身语言体系脱节；
3. 对需要 companion language 或运行时依赖的语言提供结构化元数据。

注意：
- 这是 PCE 内置的 Serena 兼容快照，不依赖运行时导入 Serena 内部 Python 模块。
- 快照应随 Serena 版本升级做显式校验和更新，而不是隐式漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

SNAPSHOT_VERSION = "serena-0.1.4-pce-snapshot-v1"


@dataclass(frozen=True)
class SerenaLanguageSpec:
    """PCE 侧的 Serena 语言快照条目。"""

    key: str
    primary_patterns: tuple[str, ...]
    activation_patterns: tuple[str, ...]
    companion_languages: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    install_channel: str | None = None
    symbol_index_enabled: bool = True
    auto_enable: bool = False
    priority: int = 100
    notes: str = ""


@dataclass(frozen=True)
class DetectedLanguageSet:
    """项目文件扫描后的语言检测结果。"""

    detected_languages: tuple[str, ...]
    representative_files: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _ts_path_patterns() -> tuple[str, ...]:
    patterns: list[str] = []
    for prefix in ("c", "m", ""):
        for postfix in ("x", ""):
            for base in ("ts", "js"):
                patterns.append(f"*.{prefix}{base}{postfix}")
    return tuple(patterns)


LANGUAGE_SPECS: tuple[SerenaLanguageSpec, ...] = (
    SerenaLanguageSpec(
        key="python",
        primary_patterns=("*.py", "*.pyi"),
        activation_patterns=("*.py", "*.pyi"),
        auto_enable=True,
        priority=10,
        notes="Pyright 由 Serena Python 运行时提供。",
    ),
    SerenaLanguageSpec(
        key="typescript",
        primary_patterns=_ts_path_patterns(),
        activation_patterns=_ts_path_patterns(),
        required_commands=("node", "npm"),
        install_channel="npm",
        auto_enable=True,
        priority=20,
        notes="Serena 会自动安装 typescript 与 typescript-language-server。",
    ),
    SerenaLanguageSpec(
        key="vue",
        primary_patterns=("*.vue",),
        activation_patterns=("*.vue",),
        companion_languages=("typescript",),
        required_commands=("node", "npm"),
        install_channel="npm",
        auto_enable=True,
        priority=30,
        notes="Vue language server 依赖 TypeScript companion language。",
    ),
    SerenaLanguageSpec(
        key="go",
        primary_patterns=("*.go",),
        activation_patterns=("*.go",),
        required_commands=("go",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=40,
        notes="通常需要 Go toolchain / gopls 环境。",
    ),
    SerenaLanguageSpec(
        key="rust",
        primary_patterns=("*.rs",),
        activation_patterns=("*.rs",),
        required_commands=("cargo", "rustc"),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=41,
        notes="通常需要 Rust toolchain / rust-analyzer 环境。",
    ),
    SerenaLanguageSpec(
        key="java",
        primary_patterns=("*.java",),
        activation_patterns=("*.java",),
        required_commands=("java",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=42,
    ),
    SerenaLanguageSpec(
        key="cpp",
        primary_patterns=("*.cpp", "*.h", "*.hpp", "*.c", "*.hxx", "*.cc", "*.cxx"),
        activation_patterns=("*.cpp", "*.h", "*.hpp", "*.c", "*.hxx", "*.cc", "*.cxx"),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=43,
    ),
    SerenaLanguageSpec(
        key="csharp",
        primary_patterns=("*.cs",),
        activation_patterns=("*.cs",),
        required_commands=("dotnet",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=44,
    ),
    SerenaLanguageSpec(
        key="kotlin",
        primary_patterns=("*.kt", "*.kts"),
        activation_patterns=("*.kt", "*.kts"),
        required_commands=("java",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=45,
    ),
    SerenaLanguageSpec(
        key="ruby",
        primary_patterns=("*.rb", "*.erb"),
        activation_patterns=("*.rb", "*.erb"),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=46,
    ),
    SerenaLanguageSpec(
        key="php",
        primary_patterns=("*.php",),
        activation_patterns=("*.php",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=47,
    ),
    SerenaLanguageSpec(
        key="swift",
        primary_patterns=("*.swift",),
        activation_patterns=("*.swift",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=48,
    ),
    SerenaLanguageSpec(
        key="terraform",
        primary_patterns=("*.tf", "*.tfvars", "*.tfstate"),
        activation_patterns=("*.tf", "*.tfvars", "*.tfstate"),
        auto_enable=True,
        priority=49,
    ),
    SerenaLanguageSpec(
        key="bash",
        primary_patterns=("*.sh", "*.bash"),
        activation_patterns=("*.sh", "*.bash"),
        required_commands=("node", "npm"),
        install_channel="npm",
        auto_enable=True,
        priority=50,
    ),
    SerenaLanguageSpec(
        key="scala",
        primary_patterns=("*.scala", "*.sbt"),
        activation_patterns=("*.scala", "*.sbt"),
        required_commands=("java",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=51,
    ),
    SerenaLanguageSpec(
        key="julia",
        primary_patterns=("*.jl",),
        activation_patterns=("*.jl",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=52,
    ),
    SerenaLanguageSpec(
        key="lua",
        primary_patterns=("*.lua",),
        activation_patterns=("*.lua",),
        auto_enable=True,
        priority=53,
    ),
    SerenaLanguageSpec(
        key="zig",
        primary_patterns=("*.zig", "*.zon"),
        activation_patterns=("*.zig", "*.zon"),
        auto_enable=True,
        priority=54,
    ),
    SerenaLanguageSpec(
        key="dart",
        primary_patterns=("*.dart",),
        activation_patterns=("*.dart",),
        auto_enable=True,
        priority=55,
    ),
    SerenaLanguageSpec(
        key="clojure",
        primary_patterns=("*.clj", "*.cljs", "*.cljc", "*.edn"),
        activation_patterns=("*.clj", "*.cljs", "*.cljc", "*.edn"),
        auto_enable=True,
        priority=56,
    ),
    SerenaLanguageSpec(
        key="elixir",
        primary_patterns=("*.ex", "*.exs"),
        activation_patterns=("*.ex", "*.exs"),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=57,
    ),
    SerenaLanguageSpec(
        key="elm",
        primary_patterns=("*.elm",),
        activation_patterns=("*.elm",),
        auto_enable=True,
        priority=58,
    ),
    SerenaLanguageSpec(
        key="erlang",
        primary_patterns=("*.erl", "*.hrl", "*.escript", "*.config", "*.app", "*.app.src"),
        activation_patterns=("*.erl", "*.hrl", "*.escript", "*.config", "*.app", "*.app.src"),
        auto_enable=True,
        priority=59,
    ),
    SerenaLanguageSpec(
        key="nix",
        primary_patterns=("*.nix",),
        activation_patterns=("*.nix",),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=60,
    ),
    SerenaLanguageSpec(
        key="ocaml",
        primary_patterns=("*.ml", "*.mli", "*.re", "*.rei"),
        activation_patterns=("*.ml", "*.mli", "*.re", "*.rei"),
        install_channel="system_toolchain",
        auto_enable=True,
        priority=61,
    ),
    SerenaLanguageSpec(
        key="al",
        primary_patterns=("*.al", "*.dal"),
        activation_patterns=("*.al", "*.dal"),
        auto_enable=True,
        priority=62,
    ),
    SerenaLanguageSpec(
        key="fsharp",
        primary_patterns=("*.fs", "*.fsx", "*.fsi"),
        activation_patterns=("*.fs", "*.fsx", "*.fsi"),
        auto_enable=True,
        priority=63,
    ),
    SerenaLanguageSpec(
        key="rego",
        primary_patterns=("*.rego",),
        activation_patterns=("*.rego",),
        auto_enable=True,
        priority=64,
    ),
    SerenaLanguageSpec(
        key="lean4",
        primary_patterns=("*.lean",),
        activation_patterns=("*.lean",),
        auto_enable=True,
        priority=65,
    ),
    SerenaLanguageSpec(
        key="fortran",
        primary_patterns=(
            "*.f90", "*.f95", "*.f03", "*.f08", "*.f", "*.for", "*.fpp",
        ),
        activation_patterns=(
            "*.f90", "*.f95", "*.f03", "*.f08", "*.f", "*.for", "*.fpp",
        ),
        auto_enable=True,
        priority=66,
    ),
    SerenaLanguageSpec(
        key="haskell",
        primary_patterns=("*.hs", "*.lhs"),
        activation_patterns=("*.hs", "*.lhs"),
        auto_enable=True,
        priority=67,
    ),
    SerenaLanguageSpec(
        key="powershell",
        primary_patterns=("*.ps1", "*.psm1", "*.psd1"),
        activation_patterns=("*.ps1", "*.psm1", "*.psd1"),
        auto_enable=True,
        priority=68,
    ),
    SerenaLanguageSpec(
        key="pascal",
        primary_patterns=("*.pas", "*.pp", "*.lpr", "*.dpr", "*.dpk", "*.inc"),
        activation_patterns=("*.pas", "*.pp", "*.lpr", "*.dpr", "*.dpk", "*.inc"),
        auto_enable=True,
        priority=69,
    ),
    SerenaLanguageSpec(
        key="matlab",
        primary_patterns=("*.m", "*.mlx", "*.mlapp"),
        activation_patterns=("*.m", "*.mlx", "*.mlapp"),
        required_commands=("node",),
        install_channel="manual",
        priority=70,
    ),
    SerenaLanguageSpec(
        key="groovy",
        primary_patterns=("*.groovy", "*.gvy"),
        activation_patterns=("*.groovy", "*.gvy"),
        auto_enable=True,
        priority=71,
    ),
)

_SPECS_BY_KEY = {spec.key: spec for spec in LANGUAGE_SPECS}


def iter_language_specs() -> tuple[SerenaLanguageSpec, ...]:
    return LANGUAGE_SPECS


def get_language_spec(language: str) -> SerenaLanguageSpec | None:
    return _SPECS_BY_KEY.get(language)


def _matches_any(patterns: tuple[str, ...], normalized_path: str) -> bool:
    path_lower = normalized_path.lower()
    return any(fnmatch(path_lower, pattern.lower()) for pattern in patterns)


def infer_language_for_path(path: str | Path) -> str | None:
    """从文件路径推断快照中的语言 key。"""
    normalized = Path(path).as_posix().lstrip("./")
    for spec in sorted(LANGUAGE_SPECS, key=lambda item: item.priority):
        if _matches_any(spec.primary_patterns, normalized):
            return spec.key
    return None


def supports_symbol_index_for_path(path: str | Path) -> bool:
    language = infer_language_for_path(path)
    if language is None:
        return False
    spec = _SPECS_BY_KEY[language]
    return spec.symbol_index_enabled


def detect_languages_from_paths(file_paths: list[str], *, max_examples: int = 3) -> DetectedLanguageSet:
    """基于本地文件列表探测项目所需语言。"""
    detected: list[str] = []
    representative_files: dict[str, list[str]] = {}

    normalized_paths = [Path(path).as_posix().lstrip("./") for path in file_paths]
    for spec in sorted(LANGUAGE_SPECS, key=lambda item: item.priority):
        matches = [path for path in normalized_paths if _matches_any(spec.activation_patterns, path)]
        if not matches:
            continue
        detected.append(spec.key)
        representative_files[spec.key] = matches[:max_examples]

    expanded = expand_languages_with_companions(detected)
    frozen_examples: dict[str, tuple[str, ...]] = {}
    for language in expanded:
        frozen_examples[language] = tuple(representative_files.get(language, []))
    return DetectedLanguageSet(
        detected_languages=tuple(expanded),
        representative_files=frozen_examples,
    )


def expand_languages_with_companions(languages: list[str] | tuple[str, ...]) -> list[str]:
    """补齐 companion languages，并保持稳定顺序。"""
    ordered = list(languages)
    changed = True
    while changed:
        changed = False
        for language in list(ordered):
            spec = _SPECS_BY_KEY.get(language)
            if spec is None:
                continue
            for companion in spec.companion_languages:
                if companion not in ordered:
                    ordered.append(companion)
                    changed = True
    return sort_language_keys(ordered)


def sort_language_keys(languages: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for language in languages:
        if language in seen:
            continue
        seen.add(language)
        normalized.append(language)
    known = [lang for lang in normalized if lang in _SPECS_BY_KEY]
    unknown = [lang for lang in normalized if lang not in _SPECS_BY_KEY]
    known.sort(key=lambda item: _SPECS_BY_KEY[item].priority)
    return [*known, *sorted(unknown)]
