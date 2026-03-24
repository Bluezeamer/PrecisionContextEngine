"""导航树核心函数单元测试（不依赖 MCP/Serena 等外部包）。"""

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 直接导入模型（不依赖 indexer.py 的 import chain）
from pce.models import AreaRecord, NavigationTree


# ---- 从 indexer.py 复制核心函数（避免 import MCP） ----

def _dedupe_keep_order(items: list) -> list:
    seen: set = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _extract_section_summary(section: dict[str, Any]) -> str:
    for line in section.get("body_lines", []):
        match = re.match(r"^职责[:：]\s*(.+)$", line.strip())
        if match and match.group(1).strip():
            return match.group(1).strip()
    for line in section.get("body_lines", []):
        stripped = line.strip()
        if not stripped or re.match(r"^(文件|详细认知)[:：]\s*", stripped):
            continue
        return stripped
    return ""


def _compute_source_digest(sections: list[dict[str, Any]]) -> str:
    payload = [
        {"slug": s["slug"], "files": sorted(_dedupe_keep_order(s.get("file_paths", [])))}
        for s in sorted(sections, key=lambda s: s["slug"])
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


NAVIGATION_FALLBACK_AREA_SLUG = "fallback"
NAVIGATION_FALLBACK_AREA_NAME = "未分类（Fallback）"


def _validate_navigation_tree(tree: NavigationTree, active_slugs: set[str]) -> list[str]:
    errors: list[str] = []
    area_slug_seen: set[str] = set()
    mounted: dict[str, str] = {}
    fallback_areas = [a for a in tree.areas if a.is_fallback]

    if not 2 <= len(tree.areas) <= 8:
        errors.append(f"area 数量必须在 2-8 之间，当前为 {len(tree.areas)}")

    for area in tree.areas:
        if area.slug in area_slug_seen:
            errors.append(f"区域 slug 重复: {area.slug}")
        area_slug_seen.add(area.slug)
        if not area.is_fallback and not area.module_slugs:
            errors.append(f"非 fallback 区域不能为空: {area.slug}")
        for ms in area.module_slugs:
            if ms not in active_slugs:
                errors.append(f"区域 {area.slug} 引用了不存在模块: {ms}")
                continue
            prev = mounted.get(ms)
            if prev is not None and prev != area.slug:
                errors.append(f"模块重复挂载: {ms} 同时出现在 {prev} / {area.slug}")
                continue
            mounted[ms] = area.slug

    if len(fallback_areas) != 1:
        errors.append(f"fallback 区域必须恰好 1 个，当前为 {len(fallback_areas)}")
    elif fallback_areas[0].slug != tree.fallback_area_slug:
        errors.append(f"fallback_area_slug 不一致")

    missing = sorted(active_slugs - set(mounted))
    if missing:
        errors.append("存在未覆盖模块: " + ", ".join(missing[:12]))
    return errors


def _repair_navigation_tree(tree: NavigationTree, active_slugs: set[str]) -> NavigationTree:
    active = set(active_slugs)
    used_modules: set[str] = set()
    used_area_slugs: set[str] = set()
    repaired_areas: list[AreaRecord] = []
    fallback_modules: list[str] = []
    fallback_order: list[str] = []
    fallback_prefixes: list[str] = []
    fallback_summary = ""

    def _uniq_slug(base: str) -> str:
        slug = base or "area"
        if slug not in used_area_slugs:
            used_area_slugs.add(slug)
            return slug
        idx = 2
        while f"{slug}-{idx}" in used_area_slugs:
            idx += 1
        uniq = f"{slug}-{idx}"
        used_area_slugs.add(uniq)
        return uniq

    for area in tree.areas:
        mods = [s for s in _dedupe_keep_order(area.module_slugs) if s in active and s not in used_modules]
        for s in mods:
            used_modules.add(s)
        order = [s for s in _dedupe_keep_order(area.recommended_order) if s in set(mods)]
        for s in mods:
            if s not in order:
                order.append(s)
        if area.is_fallback:
            fallback_modules.extend(mods)
            fallback_order.extend(order)
            fallback_prefixes.extend(area.source_prefixes)
            fallback_summary = fallback_summary or area.summary
            continue
        if not mods:
            continue
        repaired_areas.append(area.model_copy(update={
            "slug": _uniq_slug(area.slug),
            "module_slugs": mods,
            "recommended_order": order,
        }))

    missing = [s for s in sorted(active) if s not in used_modules]
    fallback_modules = _dedupe_keep_order([*fallback_modules, *missing])
    fallback_order = _dedupe_keep_order([*fallback_order, *fallback_modules])

    if len(repaired_areas) > 7:
        for overflow in repaired_areas[7:]:
            fallback_modules.extend(overflow.module_slugs)
        repaired_areas = repaired_areas[:7]
        fallback_modules = _dedupe_keep_order(fallback_modules)
        fallback_order = _dedupe_keep_order([*fallback_order, *fallback_modules])

    fb_slug = tree.fallback_area_slug or NAVIGATION_FALLBACK_AREA_SLUG
    if fb_slug in used_area_slugs:
        fb_slug = _uniq_slug(NAVIGATION_FALLBACK_AREA_SLUG)
    else:
        used_area_slugs.add(fb_slug)

    fb_area = AreaRecord(
        slug=fb_slug, display_name=NAVIGATION_FALLBACK_AREA_NAME,
        summary=fallback_summary or "承接暂时无法稳定归入主区域的模块。",
        module_slugs=fallback_modules, recommended_order=fallback_order,
        source_prefixes=_dedupe_keep_order(fallback_prefixes), is_fallback=True,
    )
    proj_summary = (tree.project_summary or "").strip()
    if not proj_summary:
        proj_summary = f"项目当前共 {len(active)} 个模块。"
    return tree.model_copy(update={
        "generated_at": datetime.now(UTC),
        "project_summary": proj_summary,
        "fallback_area_slug": fb_area.slug,
        "areas": [*repaired_areas, fb_area],
    })


def _build_fallback_navigation_tree(sections, active_slugs):
    active = set(active_slugs)
    groups: dict[str, list[str]] = defaultdict(list)
    for section in sections:
        slug = section["slug"]
        if slug not in active:
            continue
        fps = _dedupe_keep_order(section.get("file_paths", []))
        if not fps:
            groups["./"].append(slug)
            continue
        head = Path(fps[0]).parts[0] if len(Path(fps[0]).parts) > 1 else "./"
        groups[head].append(slug)

    ranked = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0]))
    areas = []
    used = set()
    used_area_slugs = set()

    def _uniq_area_slug(base):
        slug = base or "area"
        if slug not in used_area_slugs:
            used_area_slugs.add(slug)
            return slug
        idx = 2
        while f"{slug}-{idx}" in used_area_slugs:
            idx += 1
        uniq = f"{slug}-{idx}"
        used_area_slugs.add(uniq)
        return uniq

    for prefix, slugs in ranked[:7]:
        ordered = _dedupe_keep_order(slugs)
        used.update(ordered)
        name = "项目根目录" if prefix == "./" else prefix.rstrip("/").replace("-", " ").title()
        area_slug = _uniq_area_slug(name.lower().replace(" ", "-").replace("/", "-").strip("-"))
        areas.append(AreaRecord(
            slug=area_slug, display_name=name,
            summary=f"覆盖 `{prefix}` 前缀。",
            module_slugs=ordered, recommended_order=ordered,
            source_prefixes=[prefix], is_fallback=False,
        ))

    missing = [s for s in sorted(active) if s not in used]
    fb_area = AreaRecord(
        slug=NAVIGATION_FALLBACK_AREA_SLUG, display_name=NAVIGATION_FALLBACK_AREA_NAME,
        summary="承接零散模块。", module_slugs=missing, recommended_order=missing,
        source_prefixes=[], is_fallback=True,
    )
    if not areas:
        seed = sorted(active)[:8]
        areas.append(AreaRecord(
            slug="core", display_name="核心模块", summary="确定性主区域。",
            module_slugs=seed, recommended_order=seed, source_prefixes=[], is_fallback=False,
        ))
        fb_area = fb_area.model_copy(update={
            "module_slugs": [s for s in sorted(active) if s not in set(seed)],
            "recommended_order": [s for s in sorted(active) if s not in set(seed)],
        })

    return NavigationTree(
        generated_at=datetime.now(UTC), project_summary=f"共 {len(active)} 个模块。",
        fallback_area_slug=fb_area.slug, source_digest=_compute_source_digest(sections),
        areas=[*areas, fb_area],
    )


def _render_hierarchical_index_md(tree, sections_by_slug):
    total = sum(len(a.module_slugs) for a in tree.areas)
    lines = ["# 项目认知导航", "", "## 项目概览", tree.project_summary or "暂无。", "",
             f"- 区域数：{len(tree.areas)}", f"- 模块数：{total}", "", "## 区域入口"]
    for area in tree.areas:
        lines.append(f"- [{area.display_name}](areas/{area.slug}.md)：{area.summary}")
    return "\n".join(lines).rstrip() + "\n"


def _render_area_md(area, sections_by_slug):
    lines = [f"# {area.display_name}", "", "## 区域说明", area.summary or "暂无。", "", "## 模块列表"]
    for slug in area.module_slugs:
        name = sections_by_slug.get(slug, {}).get("name", slug)
        lines.append(f"- [{name}](../modules/{slug}.md)")
    lines.extend(["", "## 推荐阅读顺序"])
    ordered = area.recommended_order or area.module_slugs
    for idx, slug in enumerate(ordered, 1):
        name = sections_by_slug.get(slug, {}).get("name", slug)
        lines.append(f"{idx}. [{name}](../modules/{slug}.md)")
    return "\n".join(lines).rstrip() + "\n"


# ====================== 测试用例 ======================

def test_source_digest():
    s1 = [{"slug": "a", "file_paths": ["x.py"]}, {"slug": "b", "file_paths": ["y.py"]}]
    s2 = [{"slug": "b", "file_paths": ["y.py"]}, {"slug": "a", "file_paths": ["x.py"]}]
    assert _compute_source_digest(s1) == _compute_source_digest(s2), "digest 应对排序不变"
    s3 = [{"slug": "a", "file_paths": ["x.py", "z.py"]}, {"slug": "b", "file_paths": ["y.py"]}]
    assert _compute_source_digest(s1) != _compute_source_digest(s3), "不同文件应有不同 digest"
    print("  [PASS] test_source_digest")


def test_extract_section_summary():
    assert _extract_section_summary({"body_lines": ["职责：核心模块"]}) == "核心模块"
    assert _extract_section_summary({"body_lines": ["文件：a.py", "这是说明"]}) == "这是说明"
    assert _extract_section_summary({"body_lines": []}) == ""
    print("  [PASS] test_extract_section_summary")


def test_validate_catches_missing_modules():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        areas=[
            AreaRecord(slug="core", display_name="Core", module_slugs=["m1"]),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    errors = _validate_navigation_tree(tree, {"m1", "m2"})
    assert any("m2" in e for e in errors)
    print("  [PASS] test_validate_catches_missing_modules")


def test_validate_catches_duplicate_mount():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        areas=[
            AreaRecord(slug="a", display_name="A", module_slugs=["m1"]),
            AreaRecord(slug="b", display_name="B", module_slugs=["m1"]),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    errors = _validate_navigation_tree(tree, {"m1"})
    assert any("重复挂载" in e for e in errors)
    print("  [PASS] test_validate_catches_duplicate_mount")


def test_validate_catches_empty_non_fallback():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        areas=[
            AreaRecord(slug="empty", display_name="Empty"),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    errors = _validate_navigation_tree(tree, set())
    assert any("为空" in e for e in errors)
    print("  [PASS] test_validate_catches_empty_non_fallback")


def test_repair_adds_missing_to_fallback():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        areas=[
            AreaRecord(slug="core", display_name="Core", module_slugs=["m1"]),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    repaired = _repair_navigation_tree(tree, {"m1", "m2", "m3"})
    fb = [a for a in repaired.areas if a.is_fallback][0]
    assert "m2" in fb.module_slugs and "m3" in fb.module_slugs
    errors = _validate_navigation_tree(repaired, {"m1", "m2", "m3"})
    assert not errors, f"repaired tree should be valid: {errors}"
    print("  [PASS] test_repair_adds_missing_to_fallback")


def test_repair_removes_unknown_slugs():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        areas=[
            AreaRecord(slug="core", display_name="Core", module_slugs=["m1", "ghost"]),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    repaired = _repair_navigation_tree(tree, {"m1"})
    core = [a for a in repaired.areas if a.slug == "core"][0]
    assert "ghost" not in core.module_slugs
    print("  [PASS] test_repair_removes_unknown_slugs")


def test_fallback_tree_deterministic():
    sections = [
        {"slug": "m1", "name": "M1", "file_paths": ["src/a.py"]},
        {"slug": "m2", "name": "M2", "file_paths": ["src/b.py"]},
        {"slug": "m3", "name": "M3", "file_paths": ["tests/t.py"]},
    ]
    tree = _build_fallback_navigation_tree(sections, {"m1", "m2", "m3"})
    errors = _validate_navigation_tree(tree, {"m1", "m2", "m3"})
    assert not errors, f"fallback tree should be valid: {errors}"
    assert len(tree.areas) >= 2
    print("  [PASS] test_fallback_tree_deterministic")


def test_fallback_tree_slug_uniqueness():
    """目录名可能归一化后重复，确保 slug 去重。"""
    sections = [
        {"slug": "m1", "name": "M1", "file_paths": ["foo-bar/a.py"]},
        {"slug": "m2", "name": "M2", "file_paths": ["foo_bar/b.py"]},
        {"slug": "m3", "name": "M3", "file_paths": ["other/c.py"]},
    ]
    tree = _build_fallback_navigation_tree(sections, {"m1", "m2", "m3"})
    area_slugs = [a.slug for a in tree.areas]
    assert len(area_slugs) == len(set(area_slugs)), f"slug 重复: {area_slugs}"
    errors = _validate_navigation_tree(tree, {"m1", "m2", "m3"})
    assert not errors, f"should be valid: {errors}"
    print("  [PASS] test_fallback_tree_slug_uniqueness")


def test_render_index_has_no_module_details():
    tree = NavigationTree(
        generated_at=datetime.now(UTC), fallback_area_slug="fb",
        project_summary="Test project",
        areas=[
            AreaRecord(slug="core", display_name="Core", module_slugs=["m1"]),
            AreaRecord(slug="fb", display_name="FB", is_fallback=True),
        ],
    )
    sections_by_slug = {"m1": {"name": "M1", "body_lines": ["职责：核心逻辑", "关键符号：foo, bar"]}}
    md = _render_hierarchical_index_md(tree, sections_by_slug)
    assert "areas/core.md" in md, "should have area link"
    assert "关键符号" not in md, "should not have module details in index"
    print("  [PASS] test_render_index_has_no_module_details")


def test_render_area_no_module_prose():
    area = AreaRecord(slug="core", display_name="Core", summary="核心区域",
                      module_slugs=["m1"], recommended_order=["m1"])
    sections_by_slug = {"m1": {"name": "M1", "body_lines": ["职责：核心", "关键流程：xxx"]}}
    md = _render_area_md(area, sections_by_slug)
    assert "关键流程" not in md, "area doc should not reproduce module prose"
    assert "../modules/m1.md" in md, "should link to module doc"
    print("  [PASS] test_render_area_no_module_prose")


def test_title_compat():
    """AreaRecord 应同时支持 title= 和 display_name= 创建。"""
    a1 = AreaRecord(slug="x", title="Old Name")
    a2 = AreaRecord(slug="x", display_name="New Name")
    assert a1.display_name == "Old Name"
    assert a1.title == "Old Name"
    assert a2.display_name == "New Name"
    print("  [PASS] test_title_compat")


if __name__ == "__main__":
    print("Running navigation tree unit tests...")
    test_source_digest()
    test_extract_section_summary()
    test_validate_catches_missing_modules()
    test_validate_catches_duplicate_mount()
    test_validate_catches_empty_non_fallback()
    test_repair_adds_missing_to_fallback()
    test_repair_removes_unknown_slugs()
    test_fallback_tree_deterministic()
    test_fallback_tree_slug_uniqueness()
    test_render_index_has_no_module_details()
    test_render_area_no_module_prose()
    test_title_compat()
    print("\n=== All 12 tests passed! ===")
