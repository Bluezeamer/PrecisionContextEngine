"""
Impact 输出契约回归脚本。

目标：
1. 验证 impact prompt 已明确要求“直接调用点 / 数据契约消费者 / 间接传播链”分层输出；
2. 验证 fallback Markdown 已切换到新的 impact 结构。

运行：
    uv run python scripts/test_impact_contract.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pce.agent import (
    _build_impact_strategy_prompt,
    _build_impact_task_prompt,
    _derive_strategy_search_keys,
    _extract_strategy_search_keys,
    _normalize_impact_strategy,
    _parse_query_response,
    _parse_impact_response,
    _post_validate_markdown_locations,
    _summarize_search_hits,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_prompt_contract() -> None:
    prompt = _build_impact_task_prompt(
        target="auto_generate_layer_order",
        change_type="change_signature",
    )
    _assert("## 直接调用点" in prompt, "impact prompt 缺少“直接调用点”结构")
    _assert("## 数据契约/返回值消费者" in prompt, "impact prompt 缺少“数据契约/返回值消费者”结构")
    _assert("## 间接传播链" in prompt, "impact prompt 缺少“间接传播链”结构")
    _assert(
        "只有被工具直接确认过的定位才能写成 `path:line`" in prompt,
        "impact prompt 未强调行号必须来自工具证据",
    )
    _assert(
        "每条 bullet 只能绑定一个定位" in prompt,
        "impact prompt 未约束单条 bullet 只能绑定一个定位",
    )
    _assert(
        "在 deliver 前，应回读或检索你准备写出的每一个 `path:line`" in prompt,
        "impact prompt 未要求 deliver 前复核引用定位",
    )
    _assert(
        "不要把目标定义文件的行号误写成调用点的行号" in prompt, "impact prompt 未约束错误行号漂移"
    )


def test_strategy_prompt_contract() -> None:
    prompt = _build_impact_strategy_prompt(
        target="backend/app.py 中 /api/v2/layer-order 返回字段 layer_order",
        change_type="modify",
    )
    _assert("## 策略画像" in prompt, "策略画像 prompt 缺少固定标题")
    _assert("## 首轮证据计划" in prompt, "策略画像 prompt 缺少首轮证据计划结构")
    _assert("## 首轮检索键" in prompt, "策略画像 prompt 缺少首轮检索键结构")
    _assert(
        "task_profile: symbol-first | dataflow-first | mixed" in prompt,
        "策略画像 prompt 缺少 task_profile 约束",
    )
    _assert("tool_guidance" in prompt, "策略画像 prompt 缺少工具倾向约束")


def test_strategy_normalization_fallback() -> None:
    strategy = _normalize_impact_strategy(
        "", target="backend/app.py 中 /api/v2/layer-order 返回字段 layer_order"
    )
    _assert("## 策略画像" in strategy, "空策略未降级为默认策略块")
    _assert("task_profile: mixed" in strategy, "默认策略未回退到 mixed")
    _assert("## 首轮证据计划" in strategy, "默认策略缺少首轮证据计划")
    _assert("## 首轮检索键" in strategy, "默认策略缺少首轮检索键")


def test_strategy_normalization_appends_missing_plan() -> None:
    strategy = _normalize_impact_strategy(
        "\n".join(
            [
                "## 策略画像",
                "- task_profile: dataflow-first",
                "- primary_evidence: 先确认字段构造与接收点",
                "- secondary_evidence: 再补下游传递和消费者",
                "- tool_guidance: 先 pattern 检索，再按需读局部文件",
                "- pitfalls: 不要把字段题误收缩成纯符号题",
            ]
        ),
        target="backend/app.py 中 /api/v2/layer-order 返回字段 layer_order",
    )
    _assert("## 首轮证据计划" in strategy, "缺失计划时未自动补齐首轮证据计划")
    _assert("## 首轮检索键" in strategy, "缺失检索键时未自动补齐首轮检索键")


def test_strategy_search_keys_extract() -> None:
    strategy = "\n".join(
        [
            "## 策略画像",
            "- task_profile: dataflow-first",
            "",
            "## 首轮检索键",
            "1. layer_order",
            "2. /api/v2/layer-order",
            "3. layerOrderResult",
        ]
    )
    keys = _extract_strategy_search_keys(strategy)
    _assert(
        keys == ["layer_order", "/api/v2/layer-order", "layerOrderResult"], "首轮检索键提取异常"
    )


def test_derive_strategy_search_keys() -> None:
    keys = _derive_strategy_search_keys(
        "backend/app.py 中 /api/v2/layer-order 返回字段 layer_order"
    )
    _assert("/api/v2/layer-order" in keys, "未从 target 推导出接口路径检索键")
    _assert("layer_order" in keys, "未从 target 推导出字段检索键")


def test_summarize_search_hits() -> None:
    summary = _summarize_search_hits(
        "layer_order",
        {
            "backend/app.py": ['  > 996:        "layer_order": layer_order,'],
            "frontend/src/App.vue": [
                "  > 578:    fd.append('layer_order', JSON.stringify(layerOrderResult.value.layer_order))"
            ],
        },
    )
    _assert("搜索 `layer_order`" in summary, "搜索摘要缺少标题")
    _assert("`backend/app.py`" in summary, "搜索摘要缺少后端命中")


def test_fallback_structure() -> None:
    result = _parse_impact_response("__REACT_NO_TOOL_EXHAUSTED__")
    markdown = result.markdown
    _assert("## 直接调用点" in markdown, "fallback 缺少“直接调用点”标题")
    _assert("## 数据契约/返回值消费者" in markdown, "fallback 缺少“数据契约/返回值消费者”标题")
    _assert("## 间接传播链" in markdown, "fallback 缺少“间接传播链”标题")
    _assert("## 边界符号" in markdown, "fallback 缺少“边界符号”标题")
    _assert("## 建议修改顺序" in markdown, "fallback 缺少“建议修改顺序”标题")


def test_post_validate_shifts_to_nearby_code_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "backend" / "app.py"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(
            "\n".join(
                [
                    "def wrapper():",
                    "    # 自动生成层序",
                    "    layer_order = auto_generate_layer_order(materials=materials)",
                    "    return layer_order",
                ]
            ),
            encoding="utf-8",
        )
        markdown = "\n".join(
            [
                "## 直接调用点",
                "- `backend/app.py:2` — 在 v2_layer_order 中直接调用 auto_generate_layer_order",
            ]
        )
        normalized = _post_validate_markdown_locations(markdown, project_root=root)
        _assert("`backend/app.py:3`" in normalized, "后校验未将注释行纠偏到附近真实代码行")


def test_post_validate_downgrades_unverifiable_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "frontend" / "App.vue"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(
            "\n".join(
                [
                    "<script setup>",
                    "const status = 'idle'",
                    "</script>",
                ]
            ),
            encoding="utf-8",
        )
        markdown = "\n".join(
            [
                "## 间接传播链",
                "- `frontend/App.vue:99` — layerOrderResult.value.layer_order 被用于生成高度图时传递给 /api/v2/height-map",
            ]
        )
        normalized = _post_validate_markdown_locations(markdown, project_root=root)
        _assert("`frontend/App.vue`" in normalized, "无法复核的行号应降级为文件路径")
        _assert("`frontend/App.vue:99`" not in normalized, "无法复核的行号不应保留")


def test_post_validate_prefers_response_receive_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "frontend" / "App.vue"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(
            "\n".join(
                [
                    "const form = { layer_order: '' }",
                    "const layerOrderResult = ref(null)",
                    "const res = await fetch('/api/v2/layer-order', { method: 'POST', body: fd })",
                    "const data = await res.json()",
                    "layerOrderResult.value = data.data",
                ]
            ),
            encoding="utf-8",
        )
        markdown = "\n".join(
            [
                "## 间接传播链",
                "- `frontend/App.vue:1` — 前端接收包含 `layer_order` 的响应数据",
            ]
        )
        normalized = _post_validate_markdown_locations(markdown, project_root=root)
        _assert(
            ("`frontend/App.vue:4`" in normalized) or ("`frontend/App.vue:5`" in normalized),
            "后校验未纠偏到真正的响应接收链附近",
        )


def test_query_response_uses_location_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "backend" / "app.py"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_text(
            "\n".join(
                [
                    "def outer():",
                    "    # 包装调用",
                    "    return auto_generate_layer_order(materials=materials)",
                ]
            ),
            encoding="utf-8",
        )
        markdown = "\n".join(
            [
                "## 结论",
                "主干链路已定位。",
                "",
                "## 关键证据",
                "- `backend/app.py:2` — 这里直接调用 auto_generate_layer_order",
                "",
                "## 相关符号",
                "- `auto_generate_layer_order` — `Function` — `hicolors_logic_v2/layer_order.py:84`",
                "",
                "## 相关文件",
                "- `backend/app.py`",
                "",
                "## 不确定项",
                "- 无",
            ]
        )
        result = _parse_query_response(markdown, project_root=root)
        _assert("`backend/app.py:3`" in result.markdown, "query 响应未复用定位纠偏逻辑")


def main() -> None:
    test_prompt_contract()
    test_strategy_prompt_contract()
    test_strategy_normalization_fallback()
    test_strategy_normalization_appends_missing_plan()
    test_fallback_structure()
    test_post_validate_shifts_to_nearby_code_line()
    test_post_validate_downgrades_unverifiable_line()
    test_post_validate_prefers_response_receive_line()
    test_query_response_uses_location_validation()
    print("impact contract tests passed")


if __name__ == "__main__":
    main()
