"""PCE 索引构建模块。

负责构建三层索引:
1. structure.md — 目录职责与模块清单
2. references.json — 符号引用索引 (通过 save_index)
3. annotations/index.md + annotations/modules/*.md — 渐进式项目认知导航

构建流程:
  list_dir 扫描目录
    → 并发 get_symbols_overview 建立符号清单
    → find_referencing_symbols 建立引用图
    → LLM 生成认知导航与模块认知文档(可降级)
    → 写入 .pce/annotations/
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import litellm

from .memory import load_index, save_index
from .models import (
    BuildStats,
    FileMeta,
    IndexEntry,
    IndexSnapshot,
    ProjectMeta,
    ReferenceEdge,
    ReferenceRelation,
    SymbolKind,
    SymbolRef,
)
from .serena_client import SerenaClient, SerenaClientError
from ._env import build_litellm_model, get_completion_overrides, get_env_text

logger = logging.getLogger(__name__)

# ============================================================================
# 常量配置
# ============================================================================

CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".ts",
        ".js",
        ".tsx",
        ".jsx",
        ".go",
        ".java",
        ".rs",
        ".cpp",
        ".c",
        ".h",
    }
)

SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        ".pce",
    }
)

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".tsx": "typescriptreact",
    ".jsx": "javascriptreact",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
}

# 只对这些符号类型构建引用索引(避免过多 LLM 调用)
HIGH_LEVEL_KINDS = frozenset(
    {
        SymbolKind.CLASS,
        SymbolKind.FUNCTION,
        SymbolKind.METHOD,
        SymbolKind.MODULE,
    }
)

DEFAULT_CONCURRENCY = 10

ANNOTATIONS_DIR = "annotations"
ANNOTATIONS_INDEX_FILE = "index.md"
ANNOTATIONS_MODULES_DIR = "modules"


# ============================================================================
# 工具函数
# ============================================================================


def _normalize_tool_result(value: Any) -> Any:
    """将工具返回值统一化为 dict/list 结构。

    Serena 工具返回值经过 _jsonable 处理后可能有多种形态：
    1. 直接返回 dict/list（最理想）
    2. 单元素列表包含 JSON 字符串（旧版兼容）
    3. {'meta': ..., 'content': [{'type': 'text', 'text': <data>}]} 外壳（实测形态）
    """
    # 处理 content 外壳结构：{'meta': ..., 'content': [...]}
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            # 取第一个 type=text 的条目（兼容多元素 content）
            text_item = next(
                (item for item in content if isinstance(item, dict) and item.get("type") == "text"),
                None,
            )
            if text_item is not None:
                inner = text_item["text"]
                if isinstance(inner, str):
                    try:
                        return json.loads(inner)
                    except (ValueError, json.JSONDecodeError):
                        return inner
                return inner
    # 处理单元素列表包含 JSON 字符串的情况（旧版兼容）
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        try:
            return json.loads(value[0])
        except (ValueError, json.JSONDecodeError):
            return value[0]
    return value


def _should_skip(path: Path) -> bool:
    """判断路径是否在跳过列表中。"""
    return any(part in SKIP_DIRS for part in path.parts)


def _is_code_file(path: Path) -> bool:
    """判断路径是否为支持的代码文件。"""
    return path.suffix.lower() in CODE_EXTENSIONS


def _infer_language(path: Path) -> str:
    """从文件扩展名推断语言标识。"""
    return LANGUAGE_MAP.get(path.suffix.lower(), path.suffix.lstrip(".") or "text")


def _count_lines(path: Path) -> int:
    """统计文件行数,失败时返回 0。"""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        logger.debug(f"统计行数失败: {path}")
        return 0


def _extract_file_list(payload: Any) -> list[str]:
    """从 list_dir 响应中提取文件路径列表。"""
    if isinstance(payload, dict):
        files = payload.get("files") or payload.get("items") or []
        if isinstance(files, list):
            return [str(f) for f in files]
    if isinstance(payload, list):
        return [str(f) for f in payload if isinstance(f, (str, Path))]
    return []


def _flatten_symbols(payload: Any) -> list[dict[str, Any]]:
    """从 get_symbols_overview 响应中提取所有符号。

    Serena 的 get_symbols_overview 返回按类型分组的嵌套结构：
      {"Class": ["Foo", {"Bar": {"Method": ["baz"]}}], "Function": ["qux"]}
    键 = 符号类型, 值 = 列表, 元素为字符串(简单符号)或 dict(嵌套容器)。
    此函数将其展平为 [{"name": ..., "kind": ...}, ...] 的统一列表。
    """
    _KIND_KEYS = {
        "class",
        "interface",
        "method",
        "function",
        "module",
        "file",
        "variable",
        "import",
        "enum",
        "property",
        "constructor",
        "field",
        "constant",
        "namespace",
    }
    result: list[dict[str, Any]] = []

    def _is_kind_group(d: dict[str, Any]) -> bool:
        """判断 dict 的键是否全部为符号类型名。"""
        return bool(d) and all(str(k).lower() in _KIND_KEYS for k in d)

    def _walk(node: Any, kind: str | None) -> None:
        if isinstance(node, str):
            if kind:
                result.append({"name": node, "kind": kind})
            return
        if isinstance(node, list):
            for item in node:
                _walk(item, kind)
            return
        if isinstance(node, dict):
            # 情况 1: {"Class": [...], "Function": [...]} — 按类型分组
            if _is_kind_group(node):
                for k, v in node.items():
                    _walk(v, str(k))
                return
            # 情况 2: {"SymbolRef": {"Method": [...]}} — 嵌套容器符号
            for name, children in node.items():
                if kind:
                    result.append({"name": str(name), "kind": kind})
                _walk(children, None)

    _walk(payload, None)
    return result


def _flatten_references(payload: Any) -> list[dict[str, Any]]:
    """从 find_referencing_symbols 响应中提取所有引用 dict。"""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        result: list[dict[str, Any]] = []
        for v in payload.values():
            result.extend(_flatten_references(v))
        return result
    return []


def _map_symbol_kind(raw: Any) -> SymbolKind:
    """将 Serena 的 kind 字段映射到 SymbolKind 枚举。"""
    if raw is None:
        return SymbolKind.FUNCTION
    key = str(raw).lower().replace("symbolkind.", "")
    mapping: dict[str, SymbolKind] = {
        "class": SymbolKind.CLASS,
        "interface": SymbolKind.CLASS,
        "method": SymbolKind.METHOD,
        "function": SymbolKind.FUNCTION,
        "module": SymbolKind.MODULE,
        "file": SymbolKind.MODULE,
        "variable": SymbolKind.VARIABLE,
        "import": SymbolKind.IMPORT,
    }
    return mapping.get(key, SymbolKind.FUNCTION)


def _symbol_from_dict(payload: dict[str, Any], file_path: str) -> SymbolRef | None:
    """从符号字典构建 SymbolRef 对象。"""
    name = payload.get("name") or payload.get("name_path")
    if not name:
        return None

    # 解析行号范围
    line_start = 1
    line_end = 1
    location = payload.get("body_location") or payload.get("location") or {}
    if isinstance(location, dict):
        start = location.get("start_line") or location.get("line")
        end = location.get("end_line")
        if isinstance(start, int) and start > 0:
            line_start = start
        if isinstance(end, int) and end >= line_start:
            line_end = end
        else:
            line_end = line_start

    try:
        return SymbolRef(
            symbol_id=str(uuid.uuid4()),
            name=str(name),
            kind=_map_symbol_kind(payload.get("kind")),
            file_path=Path(file_path),
            line_start=line_start,
            line_end=line_end,
            signature=None,
        )
    except Exception as e:
        logger.debug(f"构建 SymbolRef 失败: {name}: {e}")
        return None


def _edge_from_reference(target: SymbolRef, ref: dict[str, Any]) -> ReferenceEdge:
    """从引用字典构建 ReferenceEdge 对象。"""
    ref_path = ref.get("relative_path") or ""
    location = ref.get("body_location") or {}
    line_start = location.get("start_line")

    # 构建证据字符串
    parts: list[str] = []
    if ref_path:
        parts.append(ref_path)
        if isinstance(line_start, int) and line_start > 0:
            parts.append(f":{line_start}")
    snippet = ref.get("content_around_reference") or ref.get("snippet") or ""
    if snippet:
        parts.append(f" {snippet[:80]}")  # 截断过长的片段

    evidence = "".join(parts) or target.name

    return ReferenceEdge(
        from_symbol_id=target.symbol_id,
        to_symbol_id=str(uuid.uuid4()),
        relation=ReferenceRelation.USES,
        evidence=evidence,
    )


# ============================================================================
# 原子文件写入
# ============================================================================


async def _atomic_write_text(path: Path, content: str) -> None:
    """原子写入文本文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(content)
        await asyncio.to_thread(os.replace, tmp_path, path)
    except Exception:
        if tmp_path.exists():
            await asyncio.to_thread(tmp_path.unlink, missing_ok=True)
        raise


# ============================================================================
# structure.md 生成
# ============================================================================


def _build_structure_md(entries: list[IndexEntry]) -> str:
    """生成 structure.md 文本内容。"""
    # 收集顶层目录
    top_dirs: dict[str, list[str]] = {}
    for entry in entries:
        path = Path(entry.file_meta.path)
        parts = path.parts
        top_dir = f"{parts[0]}/" if len(parts) > 1 else "./"
        top_dirs.setdefault(top_dir, []).append(path.as_posix())

    lines = ["# 项目结构索引\n", "## 目录职责"]
    for directory in sorted(top_dirs.keys()):
        file_count = len(top_dirs[directory])
        lines.append(f"- `{directory}` — {file_count} 个文件")

    lines.extend(
        [
            "",
            "## 顶层模块清单",
            "| 模块 | 文件 | 符号数 |",
            "|------|------|--------|",
        ]
    )
    for entry in sorted(entries, key=lambda e: str(e.file_meta.path)):
        path = Path(entry.file_meta.path)
        module = path.stem or path.name
        file_path = path.as_posix()
        symbol_count = len(entry.symbols)
        lines.append(f"| {module} | {file_path} | {symbol_count} |")

    return "\n".join(lines) + "\n"


async def _write_structure_md(entries: list[IndexEntry], root_path: Path) -> None:
    """将 structure.md 写入 .pce 目录。"""
    path = root_path / ".pce" / "structure.md"
    content = _build_structure_md(entries)
    await _atomic_write_text(path, content)


# ============================================================================
# 渐进式认知导航生成
# ============================================================================


def _annotation_index_path(root_path: Path) -> Path:
    """返回 .pce/annotations/index.md 路径。"""
    return root_path / ".pce" / ANNOTATIONS_DIR / ANNOTATIONS_INDEX_FILE


def _annotation_modules_dir(root_path: Path) -> Path:
    """返回 .pce/annotations/modules/ 目录路径。"""
    return root_path / ".pce" / ANNOTATIONS_DIR / ANNOTATIONS_MODULES_DIR


def _module_slug(module_name: str) -> str:
    """将模块名规范化为文件名 slug。"""
    normalized = " ".join(module_name.strip().split()) or "unnamed-module"
    return normalized.lower().replace(" ", "-")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """按出现顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _strip_markdown_fence(content: str) -> str:
    """去掉模型偶发输出的 Markdown 代码块包裹。"""
    raw = content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    return raw


def _extract_llm_text(response: Any) -> str:
    """从 litellm 响应中提取文本内容。"""
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        return getattr(msg, "content", "") or ""
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return ""


def _resolve_annotation_model(model: str | None) -> str | None:
    """解析注解生成所用的模型配置，失败时返回 None（触发降级）。"""
    if model is not None:
        return model
    provider = get_env_text("PCE_PROVIDER")
    model_name = get_env_text("PCE_MODEL")
    if not provider or not model_name:
        logger.warning("未配置 PCE_PROVIDER 或 PCE_MODEL，跳过 LLM 认知文档生成")
        return None
    return build_litellm_model(provider, model_name)


async def _llm_complete_text(
    prompt: str,
    *,
    system_prompt: str,
    model: str | None,
    failure_log: str,
) -> str | None:
    """执行单次文本补全，返回去除代码块包裹后的纯文本，失败时返回 None。"""
    effective_model = _resolve_annotation_model(model)
    if effective_model is None:
        return None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await asyncio.to_thread(
            litellm.completion,
            model=effective_model,
            messages=messages,
            temperature=0.1,
            **get_completion_overrides(),
        )
    except Exception as e:
        logger.warning(f"{failure_log}(已降级): {e}")
        return None

    content = _strip_markdown_fence(_extract_llm_text(response))
    return content.strip() or None


def _format_entry_outline(entry: IndexEntry) -> str:
    """将单个 IndexEntry 格式化为提示词摘要行（含主要符号名）。"""
    symbol_summary = ", ".join(
        f"{sym.name}({sym.kind.value})" for sym in entry.symbols[:8]
    ) or "无显式符号"
    if len(entry.symbols) > 8:
        symbol_summary += ", ..."
    return (
        f"- {entry.file_meta.path} "
        f"[{entry.file_meta.language}, {entry.file_meta.loc} 行, {len(entry.symbols)} 个符号] "
        f"符号: {symbol_summary}"
    )


def _build_index_md_prompt(entries: list[IndexEntry], project_meta: ProjectMeta) -> str:
    """构建生成 annotations/index.md 的 LLM 提示词（含 in-context learning 示例）。"""
    summary_lines = [_format_entry_outline(entry) for entry in entries[:80]]
    return "\n".join(
        [
            "你要为代码库生成一个可渐进加载的项目认知导航首页 index.md。",
            "请严格模仿示例的结构输出，不要输出任何额外解释，不要使用代码块。",
            "",
            "示例输出:",
            "# 项目认知导航",
            "",
            "## Agent Core",
            "文件：pce/agent.py, pce/agent_runtime/contracts.py, pce/agent_runtime/spawner.py",
            "职责：负责 PCE 主循环、任务编排与子 Agent 协议。对外接收查询目标，对内协调工具调用、compact 与交付流程。高风险点：ReAct 循环无 tool_calls 时的纠错逻辑、spawn 预算与深度限制。",
            "详细认知：.pce/annotations/modules/agent-core.md",
            "",
            "## Index Pipeline",
            "文件：pce/indexer.py, pce/memory.py",
            "职责：负责代码索引构建与持久化。维护结构索引、引用索引和渐进式认知文档；提供增量更新路径避免全量重建。高风险点：LLM 注解降级时需保证回退内容可机器解析。",
            "详细认知：.pce/annotations/modules/index-pipeline.md",
            "",
            f"项目根路径: {project_meta.root_path}",
            f"文件总数: {project_meta.file_count}, 代码行总数: {project_meta.loc_total}",
            "",
            "索引摘要:",
            *summary_lines,
            "",
            "输出约束:",
            "- 第一行必须是 `# 项目认知导航`",
            "- 每个模块使用 `## 模块名` 开头",
            "- `文件：` 行使用逗号分隔的相对路径，路径必须来自索引摘要",
            "- `职责：` 2-3 句话，描述模块边界、关键职责、主要协作对象和高风险点",
            "- `详细认知：` 写成 `.pce/annotations/modules/{slug}.md`，slug = 模块名小写后空格替换为连字符",
            "- 不要输出 JSON、代码块、前言、结语或任何未在示例中出现的附加文本",
        ]
    )


def _build_module_annotation_prompt(
    module_name: str,
    module_entries: list[IndexEntry],
) -> str:
    """构建单模块深度认知文档的 LLM 提示词。"""
    file_lines: list[str] = []
    for entry in module_entries:
        symbol_lines = ", ".join(
            f"{sym.name}({sym.kind.value})[{sym.line_start}-{sym.line_end}]"
            for sym in entry.symbols[:12]
        ) or "无显式符号"
        if len(entry.symbols) > 12:
            symbol_lines += ", ..."
        file_lines.append(
            f"- {entry.file_meta.path} [{entry.file_meta.language}, {entry.file_meta.loc} 行] "
            f"符号: {symbol_lines}"
        )

    return "\n".join(
        [
            "你要为单个代码模块生成可按需加载的深度认知文档。",
            "请输出 Markdown，不要代码块，不要额外解释。",
            "",
            f"模块名: {module_name}",
            f"覆盖文件数: {len(module_entries)}",
            "",
            "文件与符号摘要:",
            *file_lines,
            "",
            "输出要求:",
            f"- 标题必须是 `# {module_name}`",
            "- 必须包含 `## 覆盖文件`、`## 核心职责`、`## 关键符号`、`## 关键流程`、`## 外部协作`、`## 风险与约束` 这些二级标题",
            "- `## 覆盖文件` 需列出所有文件路径",
            "- `## 关键符号` 需优先引用给定摘要里的符号名和文件路径",
            "- `## 关键流程` 关注控制流、数据流或调用链，不要泛泛而谈",
            "- 结论必须基于输入，不得编造不存在的文件、符号或依赖",
        ]
    )


def _build_module_assignment_prompt(entry: IndexEntry, index_content: str) -> str:
    """构建新增文件模块归属判断的 LLM 提示词。"""
    return "\n".join(
        [
            "请根据现有项目认知导航，为一个新增文件判断最合适的模块归属。",
            "只能从已有模块 slug 中选择；如果无法可靠判断，则返回空字符串。",
            "",
            "现有导航:",
            index_content.strip(),
            "",
            "新增文件摘要:",
            _format_entry_outline(entry),
            "",
            "输出要求:",
            '- 只输出 JSON，例如 {"module_slug":"agent-core"}',
            '- 如果无法判断，输出 {"module_slug":""}',
            "- 不要输出 Markdown、代码块或解释",
        ]
    )


def _build_fallback_index_md(entries: list[IndexEntry]) -> str:
    """在 LLM 不可用时构建可机器解析的回退版 index.md（按顶层目录分组）。"""
    grouped: dict[str, list[IndexEntry]] = {}
    for entry in entries:
        path = Path(entry.file_meta.path)
        key = path.parts[0] if len(path.parts) > 1 else "__root__"
        grouped.setdefault(key, []).append(entry)

    lines = ["# 项目认知导航"]
    for key in sorted(grouped.keys()):
        module_entries = sorted(grouped[key], key=lambda e: str(e.file_meta.path))
        # 直接用 key 做 slug，避免来回转换引入噪声
        slug = key.lower().replace("_", "-").replace(" ", "-")
        module_name = key.replace("_", " ").replace("-", " ").title() if key != "__root__" else "Root Files"
        files = ", ".join(str(e.file_meta.path) for e in module_entries)
        lines.extend(
            [
                "",
                f"## {module_name}",
                f"文件：{files}",
                "职责：当前为基于目录结构的回退导航，建议模型可用时重新生成精细化认知边界。",
                f"详细认知：.pce/annotations/modules/{slug}.md",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_fallback_module_md(module_name: str, module_entries: list[IndexEntry]) -> str:
    """在 LLM 不可用时构建模块文档回退内容（基于符号索引）。"""
    lines = [
        f"# {module_name}",
        "",
        "## 覆盖文件",
    ]
    for entry in module_entries:
        lines.append(f"- {entry.file_meta.path}")

    lines.extend(["", "## 核心职责", "当前为基于索引条目的回退摘要，提供基本文件边界与符号概览。", "", "## 关键符号"])
    has_symbols = False
    for entry in module_entries:
        for sym in entry.symbols[:12]:
            lines.append(f"- {sym.name} ({sym.kind.value}) — {sym.file_path}:{sym.line_start}")
            has_symbols = True
    if not has_symbols:
        lines.append("- 无显式符号")

    lines.extend(
        [
            "",
            "## 关键流程",
            "需结合具体文件内容进一步检索，当前回退摘要不直接推断控制流。",
            "",
            "## 外部协作",
            "可结合 structure.md、references.json 及 Serena 工具继续定位模块交互关系。",
            "",
            "## 风险与约束",
            "本文件由索引级回退逻辑生成，细粒度职责与边界仍应以 LLM 生成版本为准。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


# ============================================================================
# index.md 解析与增量更新辅助
# ============================================================================


def _parse_index_section(module_name: str, body_lines: list[str]) -> dict[str, Any]:
    """将 index.md 单个模块章节解析为结构化记录。"""
    file_paths: list[str] = []
    for line in body_lines:
        m = re.match(r"^文件[:：]\s*(.+)$", line.strip())
        if m:
            file_paths = [p.strip() for p in m.group(1).split(",") if p.strip()]
            break
    return {
        "name": module_name,
        "slug": _module_slug(module_name),
        "body_lines": body_lines[:],
        "file_paths": file_paths,
    }


def _split_index_md(index_content: str) -> tuple[str, list[dict[str, Any]]]:
    """拆分 index.md 为头部文本和模块章节列表。"""
    header_lines: list[str] = []
    sections: list[dict[str, Any]] = []
    current_name: str | None = None
    current_body: list[str] = []
    seen_section = False

    for raw_line in index_content.splitlines():
        if raw_line.startswith("## "):
            seen_section = True
            if current_name is not None:
                sections.append(_parse_index_section(current_name, current_body))
            current_name = raw_line[3:].strip()
            current_body = []
            continue
        if not seen_section:
            header_lines.append(raw_line)
        elif current_name is not None:
            current_body.append(raw_line)

    if current_name is not None:
        sections.append(_parse_index_section(current_name, current_body))

    header = "\n".join(header_lines).strip() or "# 项目认知导航"
    return header, sections


def _render_index_md(header: str, sections: list[dict[str, Any]]) -> str:
    """将结构化章节列表渲染回 index.md 字符串。"""
    lines = [header.strip() or "# 项目认知导航"]
    for section in sections:
        lines.extend(["", f"## {section['name']}"])
        lines.extend(section["body_lines"])
    return "\n".join(lines).rstrip() + "\n"


def _parse_index_md(index_content: str) -> dict[str, list[str]]:
    """返回模块 slug -> 文件路径列表的映射（供增量更新反查）。"""
    _, sections = _split_index_md(index_content)
    return {
        section["slug"]: _dedupe_keep_order(section["file_paths"])
        for section in sections
        if section["file_paths"]
    }


def _update_section_file_list(section: dict[str, Any], file_paths: list[str]) -> None:
    """就地替换章节中的 `文件：` 行，并更新 file_paths 字段。"""
    normalized = _dedupe_keep_order(file_paths)
    file_line = f"文件：{', '.join(normalized)}"
    new_body: list[str] = []
    replaced = False
    for line in section["body_lines"]:
        if not replaced and re.match(r"^文件[:：]\s*", line.strip()):
            new_body.append(file_line)
            replaced = True
        else:
            new_body.append(line)
    if not replaced:
        new_body.insert(0, file_line)
    section["body_lines"] = new_body
    section["file_paths"] = normalized


def _build_temporary_section(file_path: str) -> dict[str, Any]:
    """为无法自动归属的新增文件创建临时模块章节。"""
    stem = Path(file_path).stem.replace(".", "-")
    module_name = f"临时归类 {stem}"
    slug = _module_slug(module_name)
    return {
        "name": module_name,
        "slug": slug,
        "file_paths": [file_path],
        "body_lines": [
            f"文件：{file_path}",
            "职责：该章节由增量索引降级逻辑创建，用于暂存未完成归类的新文件。建议后续执行全量索引校正模块边界。",
            f"详细认知：.pce/annotations/modules/{slug}.md",
        ],
    }


def _prepare_module_specs(
    sections: list[dict[str, Any]],
    entries_map: dict[str, "IndexEntry"],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, list["IndexEntry"]]]]:
    """过滤章节中实际存在的文件，返回有效章节列表和模块生成规格。"""
    valid_sections: list[dict[str, Any]] = []
    module_specs: list[tuple[str, str, list[IndexEntry]]] = []
    for section in sections:
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            logger.warning(f"认知导航章节未命中实际文件，跳过: {section['name']}")
            continue
        _update_section_file_list(section, file_paths)
        valid_sections.append(section)
        module_specs.append(
            (section["name"], section["slug"], [entries_map[p] for p in file_paths])
        )
    return valid_sections, module_specs


async def _generate_index_md(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    model: str | None = None,
) -> str | None:
    """调用 LLM 生成 annotations/index.md。"""
    if not entries:
        return None
    return await _llm_complete_text(
        _build_index_md_prompt(entries, project_meta),
        system_prompt="你是资深软件架构分析助手，擅长把大型代码库整理成可渐进加载的认知导航。",
        model=model,
        failure_log="认知导航 index.md 生成失败",
    )


async def _generate_module_annotation(
    module_name: str,
    module_entries: list[IndexEntry],
    model: str | None = None,
) -> str | None:
    """调用 LLM 生成单模块深度认知文档。"""
    if not module_entries:
        return None
    return await _llm_complete_text(
        _build_module_annotation_prompt(module_name, module_entries),
        system_prompt="你是资深软件架构分析助手，擅长生成可按需加载的模块深度认知文档。",
        model=model,
        failure_log=f"模块认知文档生成失败: {module_name}",
    )


async def _classify_new_entry_module(
    entry: IndexEntry,
    index_content: str,
    known_slugs: set[str],
    model: str | None = None,
) -> str | None:
    """用 LLM 判断新增文件应归属的模块 slug，不在 known_slugs 中时返回 None。"""
    if not known_slugs:
        return None
    content = await _llm_complete_text(
        _build_module_assignment_prompt(entry, index_content),
        system_prompt="你是代码库模块归类助手，只能从现有模块中选择最匹配的归属。",
        model=model,
        failure_log=f"新增文件模块归属判断失败: {entry.file_meta.path}",
    )
    if not content:
        return None
    slug = ""
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            slug = str(payload.get("module_slug") or "").strip()
    except json.JSONDecodeError:
        slug = content.strip()
    return slug if slug in known_slugs else None


async def _write_module_files(
    module_specs: list[tuple[str, str, list[IndexEntry]]],
    modules_dir: Path,
    model: str | None,
) -> None:
    """并发生成并写入 modules/*.md，失败时使用回退内容。"""
    results = await asyncio.gather(
        *[_generate_module_annotation(name, entries, model=model) for name, _, entries in module_specs],
        return_exceptions=True,
    )
    for (module_name, slug, module_entries), result in zip(module_specs, results):
        if isinstance(result, BaseException):
            logger.warning(f"模块认知文档生成异常，使用回退内容: {module_name}: {result}")
            content = None
        else:
            content = result
        if not content:
            content = _build_fallback_module_md(module_name, module_entries)
        await _atomic_write_text(modules_dir / f"{slug}.md", content.rstrip() + "\n")


async def _write_annotations(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    model: str | None = None,
) -> None:
    """全量生成并写入 annotations/index.md 与 modules/*.md。"""
    index_path = _annotation_index_path(root_path)
    modules_dir = _annotation_modules_dir(root_path)

    if not entries:
        await _atomic_write_text(index_path, "# 项目认知导航\n")
        if modules_dir.exists():
            for stale in modules_dir.glob("*.md"):
                await asyncio.to_thread(stale.unlink, missing_ok=True)
        return

    index_content = await _generate_index_md(entries, project_meta, model=model)
    if not index_content:
        logger.warning("认知导航 index.md 不可用，使用索引级回退模板")
        index_content = _build_fallback_index_md(entries)

    entries_map = {str(e.file_meta.path): e for e in entries}
    header, sections = _split_index_md(index_content)
    valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    if not valid_sections:
        logger.warning("认知导航解析为空，回退到结构化模板")
        index_content = _build_fallback_index_md(entries)
        header, sections = _split_index_md(index_content)
        valid_sections, module_specs = _prepare_module_specs(sections, entries_map)

    await _write_module_files(module_specs, modules_dir, model)

    # 删除本次不再使用的旧 module 文件
    expected = {f"{slug}.md" for _, slug, _ in module_specs}
    if modules_dir.exists():
        for stale in modules_dir.glob("*.md"):
            if stale.name not in expected:
                await asyncio.to_thread(stale.unlink, missing_ok=True)

    await _atomic_write_text(index_path, _render_index_md(header, valid_sections))
    logger.info(
        f"认知导航写入完成: {len(valid_sections)} 个模块章节, {len(module_specs)} 个模块文档"
    )


async def _update_annotations_incremental(
    entries: list[IndexEntry],
    project_meta: ProjectMeta,
    root_path: Path,
    *,
    changed_files: list[str],
    deleted_files: list[str],
    model: str | None = None,
) -> None:
    """增量更新受影响模块的认知文档与 index.md。"""
    index_path = _annotation_index_path(root_path)
    modules_dir = _annotation_modules_dir(root_path)

    try:
        async with aiofiles.open(index_path, "r", encoding="utf-8") as f:
            index_content = await f.read()
    except FileNotFoundError:
        logger.info("未找到认知导航 index.md，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return
    except Exception as e:
        logger.warning(f"读取认知导航失败，降级为全量认知文档重建: {e}")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    header, sections = _split_index_md(index_content)
    if not sections:
        logger.warning("认知导航解析结果为空，降级为全量认知文档重建")
        await _write_annotations(entries, project_meta, root_path, model=model)
        return

    entries_map = {str(e.file_meta.path): e for e in entries}
    ownership = _parse_index_md(index_content)
    reverse_map: dict[str, str] = {
        file_path: slug
        for slug, file_paths in ownership.items()
        for file_path in file_paths
    }
    sections_by_slug = {section["slug"]: section for section in sections}
    affected_slugs: set[str] = set()

    # 已知文件的变更/删除 → 标记所属模块为受影响
    for file_path in (*changed_files, *deleted_files):
        slug = reverse_map.get(file_path)
        if slug:
            affected_slugs.add(slug)

    # 新增文件（不在任何已知模块中）→ 尝试归属或创建临时章节
    for file_path in changed_files:
        if file_path in reverse_map:
            continue
        entry = entries_map.get(file_path)
        if entry is None:
            continue

        current_index_content = _render_index_md(header, sections)
        assigned_slug = await _classify_new_entry_module(
            entry, current_index_content, set(sections_by_slug.keys()), model=model
        )
        if assigned_slug and assigned_slug in sections_by_slug:
            section = sections_by_slug[assigned_slug]
            _update_section_file_list(section, section["file_paths"] + [file_path])
            affected_slugs.add(assigned_slug)
            logger.info(f"新增文件已归属既有模块: {file_path} -> {assigned_slug}")
        else:
            temp_section = _build_temporary_section(file_path)
            sections.append(temp_section)
            sections_by_slug[temp_section["slug"]] = temp_section
            affected_slugs.add(temp_section["slug"])
            logger.warning(f"新增文件归属判断失败，已追加临时章节: {file_path}")

    if not affected_slugs:
        logger.info("认知导航无受影响模块，跳过 annotations 增量更新")
        return

    # 重新生成受影响模块；无文件的模块移除
    final_sections: list[dict[str, Any]] = []
    module_specs: list[tuple[str, str, list[IndexEntry]]] = []
    removed_slugs: set[str] = set()

    for section in sections:
        slug = section["slug"]
        if slug not in affected_slugs:
            final_sections.append(section)
            continue
        file_paths = [p for p in _dedupe_keep_order(section["file_paths"]) if p in entries_map]
        if not file_paths:
            removed_slugs.add(slug)
            logger.info(f"模块已无文件，移除认知章节: {section['name']}")
            continue
        _update_section_file_list(section, file_paths)
        final_sections.append(section)
        module_specs.append((section["name"], slug, [entries_map[p] for p in file_paths]))

    await _write_module_files(module_specs, modules_dir, model)

    for slug in removed_slugs:
        module_path = modules_dir / f"{slug}.md"
        if module_path.exists():
            await asyncio.to_thread(module_path.unlink, missing_ok=True)

    await _atomic_write_text(index_path, _render_index_md(header, final_sections))
    logger.info(
        f"认知导航增量更新完成: {len(module_specs)} 个模块重建, {len(removed_slugs)} 个模块移除"
    )


# ============================================================================
# 核心索引逻辑
# ============================================================================


async def _scan_directory(serena_client: SerenaClient) -> list[str]:
    """递归扫描项目目录,返回源代码文件路径列表。"""
    try:
        raw = await serena_client.list_dir(".", recursive=True, skip_ignored_files=True)
    except SerenaClientError as e:
        logger.error(f"目录扫描失败: {e}")
        return []

    payload = _normalize_tool_result(raw)
    files = _extract_file_list(payload)

    results: list[str] = []
    for item in files:
        path = Path(item)
        if _should_skip(path) or not _is_code_file(path):
            continue
        results.append(path.as_posix())

    logger.info(f"扫描完成: 发现 {len(results)} 个源代码文件")
    return sorted(set(results))


async def _resolve_name_path(
    serena_client: SerenaClient, symbol: SymbolRef, file_path: str
) -> str | None:
    """解析符号的 name_path(Serena 格式,如 MyClass/my_method)。"""
    try:
        raw = await serena_client.find_symbol(
            symbol.name,
            relative_path=file_path,
            include_body=False,
            depth=0,
            substring_matching=False,
        )
    except SerenaClientError:
        return None

    payload = _normalize_tool_result(raw)
    for item in _flatten_symbols(payload):
        name_path = item.get("name_path") or item.get("namePath")
        if name_path:
            return str(name_path)
    return None


async def _index_file(file_path: str, serena_client: SerenaClient) -> IndexEntry | None:
    """索引单个文件,返回 IndexEntry。

    Args:
        file_path: 相对于项目根目录的文件路径
        serena_client: 已连接的 SerenaClient 实例

    Returns:
        索引条目,如果文件不可访问则返回 None
    """
    root_path = serena_client.project_path
    abs_path = root_path / file_path

    if not abs_path.exists():
        logger.warning(f"文件不存在,跳过: {file_path}")
        return None

    # 获取符号概览
    try:
        overview_raw = await serena_client.get_symbols_overview(file_path, depth=1)
    except SerenaClientError as e:
        logger.warning(f"获取符号概览失败: {file_path}: {e}")
        return None

    overview = _normalize_tool_result(overview_raw)
    symbols: list[SymbolRef] = []
    for sym_dict in _flatten_symbols(overview):
        sym = _symbol_from_dict(sym_dict, file_path)
        if sym is not None:
            symbols.append(sym)

    # 对高层符号建立引用索引
    edges: list[ReferenceEdge] = []
    for symbol in symbols:
        if symbol.kind not in HIGH_LEVEL_KINDS:
            continue

        name_path = await _resolve_name_path(serena_client, symbol, file_path)
        if not name_path:
            continue

        try:
            refs_raw = await serena_client.find_referencing_symbols(
                name_path=name_path, relative_path=file_path
            )
        except SerenaClientError:
            continue

        refs_payload = _normalize_tool_result(refs_raw)
        for ref in _flatten_references(refs_payload):
            edges.append(_edge_from_reference(symbol, ref))

    # 收集文件统计信息
    try:
        stat = abs_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        size_bytes = stat.st_size
    except Exception:
        mtime = datetime.now(timezone.utc)
        size_bytes = 0

    return IndexEntry(
        file_meta=FileMeta(
            path=Path(file_path),
            language=_infer_language(abs_path),
            size_bytes=size_bytes,
            mtime=mtime,
            loc=_count_lines(abs_path),
        ),
        symbols=symbols,
        imports=[],
        edges=edges,
    )


# ============================================================================
# 公开 API
# ============================================================================


async def build_index(
    project_path: str | Path,
    serena_client: SerenaClient,
    memory_root: str | Path | None = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str | None = None,
) -> IndexSnapshot:
    """构建三层索引并写入 Memory。

    Args:
        project_path: 目标项目根路径
        serena_client: 已连接的 Serena 客户端
        memory_root: Memory 文件的写入根路径,默认与 project_path 相同
        concurrency: 并发索引文件数上限
        model: 用于生成语义注解的 LLM 模型名称(litellm 格式)

    Returns:
        构建完成的 IndexSnapshot

    Raises:
        Exception: 索引构建失败
    """
    root_path = Path(project_path).resolve()
    memory_root_path = Path(memory_root).resolve() if memory_root else root_path
    start_time = time.monotonic()

    logger.info(f"开始构建索引: {root_path}")

    # 扫描文件列表
    files = await _scan_directory(serena_client)
    if not files:
        logger.warning("未发现任何源代码文件")

    # 并发索引所有文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    # return_exceptions=True 确保单文件异常不会中断整体构建
    results = await asyncio.gather(*[_run_with_semaphore(f) for f in files], return_exceptions=True)

    entries: list[IndexEntry] = []
    failed_files: list[str] = []
    for file_path, result in zip(files, results):
        if isinstance(result, BaseException):
            logger.warning(f"文件索引异常: {file_path}: {result}")
            failed_files.append(file_path)
        elif result is None:
            failed_files.append(file_path)
        else:
            entries.append(result)
    warnings = [f"索引失败: {path}" for path in failed_files]

    if warnings:
        logger.warning(f"有 {len(warnings)} 个文件索引失败")

    # 构建元数据
    created_at = datetime.now(timezone.utc)
    project_meta = ProjectMeta(
        root_path=root_path,
        created_at=created_at,
        index_version="1",
        file_count=len(entries),
        loc_total=sum(e.file_meta.loc for e in entries),
    )

    build_stats = BuildStats(
        total_files=len(entries),
        total_symbols=sum(len(e.symbols) for e in entries),
        total_edges=sum(len(e.edges) for e in entries),
        duration_ms=int((time.monotonic() - start_time) * 1000),
        warnings=warnings,
    )

    snapshot = IndexSnapshot(
        project_meta=project_meta,
        entries=entries,
        created_at=created_at,
        build_stats=build_stats,
    )

    # 写入 Memory
    await save_index(snapshot, root_path=memory_root_path)
    await _write_structure_md(entries, memory_root_path)

    logger.info(
        f"索引构建完成: {len(entries)} 个文件, "
        f"{build_stats.total_symbols} 个符号, "
        f"{build_stats.total_edges} 条引用边, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # 渐进式认知导航(可降级)
    try:
        await _write_annotations(entries, project_meta, memory_root_path, model=model)
    except Exception as e:
        logger.warning(f"写入项目认知导航失败(已降级): {e}")

    return snapshot


async def build_index_incremental(
    project_path: str | Path,
    serena_client: SerenaClient,
    memory_root: str | Path | None = None,
    *,
    changed_files: list[str],
    deleted_files: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str | None = None,
) -> IndexSnapshot:
    """增量索引更新：仅重建变更文件，合并到已有索引。

    若不存在历史索引，自动降级为全量 build_index。

    Args:
        project_path: 目标项目根路径
        serena_client: 已连接的 Serena 客户端
        memory_root: Memory 文件的写入根路径
        changed_files: 变更（含新增）的文件相对路径列表
        deleted_files: 已删除的文件相对路径列表
        concurrency: 并发索引文件数上限

    Returns:
        合并后的 IndexSnapshot
    """
    root_path = Path(project_path).resolve()
    memory_root_path = Path(memory_root).resolve() if memory_root else root_path
    start_time = time.monotonic()

    # 尝试加载已有索引
    existing = await load_index(root_path=memory_root_path)
    if existing is None:
        logger.info("无历史索引，降级为全量构建")
        return await build_index(
            project_path=project_path,
            serena_client=serena_client,
            memory_root=memory_root,
            concurrency=concurrency,
            model=model,
        )

    # 过滤有效的代码文件
    effective_changes = [
        f for f in changed_files if _is_code_file(Path(f)) and not _should_skip(Path(f))
    ]
    deleted = set(deleted_files or [])

    if not effective_changes and not deleted:
        logger.info("无有效变更，跳过增量更新")
        return existing

    logger.info(f"开始增量索引: {len(effective_changes)} 个变更, " f"{len(deleted)} 个删除")

    # 并发重建变更文件
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(file_path: str) -> IndexEntry | None:
        async with semaphore:
            return await _index_file(file_path, serena_client)

    results = await asyncio.gather(*[_run(f) for f in effective_changes], return_exceptions=True)

    # 以现有条目为基础，按文件路径建立映射
    entries_map: dict[str, IndexEntry] = {str(e.file_meta.path): e for e in existing.entries}

    failed_files: list[str] = []
    for file_path, result in zip(effective_changes, results):
        if isinstance(result, BaseException):
            logger.warning(f"增量索引异常: {file_path}: {result}")
            failed_files.append(file_path)
        elif result is None:
            failed_files.append(file_path)
        else:
            entries_map[str(result.file_meta.path)] = result

    # 移除已删除文件的条目
    for path in deleted:
        entries_map.pop(path, None)

    merged_entries = list(entries_map.values())
    warnings = [f"增量索引失败: {path}" for path in failed_files]

    # 更新元数据（保留原始 created_at 和 index_version）
    project_meta = ProjectMeta(
        root_path=root_path,
        created_at=existing.project_meta.created_at,
        index_version=existing.project_meta.index_version,
        file_count=len(merged_entries),
        loc_total=sum(e.file_meta.loc for e in merged_entries),
    )

    build_stats = BuildStats(
        total_files=len(merged_entries),
        total_symbols=sum(len(e.symbols) for e in merged_entries),
        total_edges=sum(len(e.edges) for e in merged_entries),
        duration_ms=int((time.monotonic() - start_time) * 1000),
        warnings=warnings,
    )

    snapshot = IndexSnapshot(
        project_meta=project_meta,
        entries=merged_entries,
        created_at=datetime.now(timezone.utc),
        build_stats=build_stats,
    )

    # 写入 Memory
    await save_index(snapshot, root_path=memory_root_path)
    await _write_structure_md(merged_entries, memory_root_path)

    logger.info(
        f"增量索引完成: {len(effective_changes)} 文件更新, "
        f"{len(deleted)} 文件删除, "
        f"合计 {len(merged_entries)} 文件, "
        f"耗时 {build_stats.duration_ms}ms"
    )

    # 增量更新认知导航(可降级)
    try:
        await _update_annotations_incremental(
            merged_entries,
            project_meta,
            memory_root_path,
            changed_files=effective_changes,
            deleted_files=sorted(deleted),
            model=model,
        )
    except Exception as e:
        logger.warning(f"增量更新项目认知导航失败(已降级): {e}")

    return snapshot
