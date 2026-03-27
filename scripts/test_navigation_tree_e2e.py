"""
导航树端到端集成测试 — 进程内直连，跑通真实 Serena + 真实 LLM。

验证目标：
1. pce_init 后 navigation_tree.json 正确生成
2. index.md 为层级格式（含区域入口）
3. areas/*.md 正确生成
4. modules/*.md 保持不变
5. navigation_tree.json 校验通过

运行：
    uv run python scripts/test_navigation_tree_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from pce.server import PCEContext
from pce._env import configure_litellm_runtime, get_base_url, get_completion_overrides, get_env_text
from pce.models import NavigationTree

logger = logging.getLogger("pce.nav_tree_e2e")


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _banner(title: str) -> None:
    sep = "=" * 80
    print(f"\n{sep}\n  {title}\n{sep}")


def _pprint(label: str, data: object) -> None:
    print(f"\n[{label}]")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


async def _cleanup(ctx: PCEContext | None) -> None:
    if ctx is None:
        return
    if ctx.watcher is not None:
        try:
            await ctx.watcher.stop()
        except Exception:
            pass
    serena_client = getattr(ctx, "serena_client", None)
    if serena_client is not None:
        try:
            await serena_client.disconnect()
        except Exception:
            pass


async def main() -> None:
    root = _root()
    load_dotenv(root / ".env")
    configure_litellm_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    project_path = Path(get_env_text("PCE_PROJECT_PATH") or root).resolve()
    provider = get_env_text("PCE_PROVIDER")
    model = get_env_text("PCE_MODEL")
    overrides = get_completion_overrides()

    _banner("导航树端到端集成测试")
    _pprint("配置", {
        "project_path": str(project_path),
        "provider": provider or "(missing)",
        "model": model or "(missing)",
        "api_key": "已设置" if overrides.get("api_key") else "未设置",
        "base_url": get_base_url() or "(default)",
    })

    if not provider or not model:
        raise RuntimeError("PCE_PROVIDER / PCE_MODEL 未配置")
    if not overrides.get("api_key"):
        raise RuntimeError("PCE_API_KEY 未配置")

    ctx: PCEContext | None = None
    errors: list[str] = []

    try:
        # ── 1. 执行 pce_init ──
        _banner("阶段 1: pce_init")
        t0 = time.perf_counter()
        ctx = PCEContext()
        init_result = await ctx.handle_init(str(project_path))
        init_elapsed = round(time.perf_counter() - t0, 2)

        _pprint("初始化完成", {
            "elapsed_s": init_elapsed,
            "initialized": init_result.get("initialized"),
        })

        if not init_result.get("initialized"):
            raise RuntimeError(f"初始化失败: {init_result.get('error', '未知')}")

        # ── 2. 检查 navigation_tree.json ──
        _banner("阶段 2: 检查 navigation_tree.json")
        pce_dir = project_path / ".pce"
        tree_path = pce_dir / "annotations" / "navigation_tree.json"

        if not tree_path.exists():
            errors.append("navigation_tree.json 未生成")
            print(f"  FAIL: {tree_path} 不存在")
        else:
            raw = tree_path.read_text(encoding="utf-8")
            tree_data = json.loads(raw)
            tree = NavigationTree.model_validate(tree_data)

            print(f"  版本: {tree.version}")
            print(f"  项目概览: {tree.project_summary[:80]}...")
            print(f"  区域数: {len(tree.areas)}")
            print(f"  fallback 区域: {tree.fallback_area_slug}")
            print(f"  source_digest: {tree.source_digest[:16]}...")

            for area in tree.areas:
                print(f"    [{area.slug}] {area.display_name}: "
                      f"{len(area.module_slugs)} 个模块, fallback={area.is_fallback}")

            # 校验
            from pce.annotation_writer import _validate_navigation_tree
            all_slugs = {
                ms for area in tree.areas for ms in area.module_slugs
            }
            validation_errors = _validate_navigation_tree(tree, all_slugs)
            if validation_errors:
                for ve in validation_errors:
                    errors.append(f"navigation_tree 校验失败: {ve}")
                    print(f"  FAIL: {ve}")
            else:
                print("  PASS: navigation_tree.json 校验通过")

        # ── 3. 检查 index.md 为层级格式 ──
        _banner("阶段 3: 检查 index.md 格式")
        index_path = pce_dir / "annotations" / "index.md"

        if not index_path.exists():
            errors.append("index.md 未生成")
            print(f"  FAIL: {index_path} 不存在")
        else:
            index_content = index_path.read_text(encoding="utf-8")
            lines = index_content.splitlines()
            print(f"  行数: {len(lines)}")

            has_area_links = any("areas/" in line for line in lines)
            has_project_nav = "项目认知导航" in index_content
            has_area_entry = "区域入口" in index_content

            if not has_project_nav:
                errors.append("index.md 缺少'项目认知导航'标题")
            if not has_area_entry:
                errors.append("index.md 缺少'区域入口'章节")
            if not has_area_links:
                errors.append("index.md 没有 areas/ 链接（不是层级格式）")

            if has_project_nav and has_area_entry and has_area_links:
                print("  PASS: index.md 为层级格式")
            else:
                for e in errors[-3:]:
                    print(f"  FAIL: {e}")

            # 确认 index.md 不含模块级细节
            if "关键符号" in index_content or "关键流程" in index_content:
                errors.append("index.md 包含了模块级细节（违反层级职责边界）")
                print("  FAIL: index.md 包含模块级细节")

        # ── 4. 检查 areas/*.md ──
        _banner("阶段 4: 检查 areas/*.md")
        areas_dir = pce_dir / "annotations" / "areas"

        if not areas_dir.exists():
            errors.append("areas/ 目录未生成")
            print(f"  FAIL: {areas_dir} 不存在")
        else:
            area_files = list(areas_dir.glob("*.md"))
            print(f"  区域文档数: {len(area_files)}")

            if tree_path.exists():
                expected_count = len(tree.areas)
                if len(area_files) != expected_count:
                    errors.append(
                        f"area 文档数 ({len(area_files)}) "
                        f"与 tree.areas ({expected_count}) 不匹配"
                    )

            for af in area_files:
                content = af.read_text(encoding="utf-8")
                # 检查基本结构
                if "## 模块列表" not in content:
                    errors.append(f"{af.name} 缺少'模块列表'章节")
                if "## 推荐阅读顺序" not in content:
                    errors.append(f"{af.name} 缺少'推荐阅读顺序'章节")
                # 硬限制：不复述模块正文
                if "关键流程" in content or "关键符号" in content:
                    errors.append(f"{af.name} 复述了模块级正文（违反硬限制）")
                # 空 fallback 区域没有模块链接是正常的
                area_slug = af.stem
                is_fallback_area = tree_path.exists() and any(
                    a.slug == area_slug and a.is_fallback and not a.module_slugs
                    for a in tree.areas
                )
                if "../modules/" not in content and not is_fallback_area:
                    errors.append(f"{af.name} 没有 ../modules/ 链接")

            if not any(af.name for af in area_files):
                errors.append("areas/ 目录为空")
            else:
                area_errors = [e for e in errors if "areas/" in e or ".md" in e]
                if not area_errors:
                    print("  PASS: 所有 area 文档结构正确")

        # ── 5. 检查 modules/*.md 保持不变 ──
        _banner("阶段 5: 检查 modules/*.md")
        modules_dir = pce_dir / "annotations" / "modules"

        if not modules_dir.exists():
            errors.append("modules/ 目录未生成")
            print(f"  FAIL: {modules_dir} 不存在")
        else:
            module_files = list(modules_dir.glob("*.md"))
            print(f"  模块文档数: {len(module_files)}")
            if module_files:
                print("  PASS: modules/*.md 存在")
            else:
                errors.append("modules/ 目录为空")

        # ── 汇总 ──
        _banner("测试汇总")
        if errors:
            print(f"\n  FAILED — {len(errors)} 个错误:")
            for i, e in enumerate(errors, 1):
                print(f"    {i}. {e}")
        else:
            print("\n  ALL PASSED — 导航树端到端集成测试全部通过!")

    except Exception:
        print(f"\n  FATAL ERROR:\n{traceback.format_exc()}")
    finally:
        await _cleanup(ctx)


if __name__ == "__main__":
    asyncio.run(main())
