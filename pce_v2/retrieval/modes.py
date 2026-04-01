from __future__ import annotations

from ..contracts import BudgetPolicy, ModeName, PromptContract
from ..tools.specs import ToolAssemblyPolicy, ToolSpec


class QueryMode:
    name = ModeName.QUERY
    budget = BudgetPolicy(
        max_tool_calls=12,
        max_reads=8,
        max_escalations=2,
        max_result_chars=8000,
    )
    contract = PromptContract(
        identity="你是 PCE v2 的 query 检索执行器。",
        objective="快速收敛入口、关键文件、关键符号与主调用链附近的高价值证据。",
        stop_when=[
            "已经能够定位主要入口或定义点",
            "已经给出足够支持上层继续工作的最小证据集",
        ],
        output_sections=["结论", "关键证据", "相关符号", "相关文件", "不确定项"],
    )
    tool_policy = ToolAssemblyPolicy(
        mode="query",
        allowed_tools=["navigation_read", "insight_read", "code_search", "code_read", "deliver"],
        denied_path_prefixes=[".pce/annotations"],
    )
    tool_specs = [
        ToolSpec(name="navigation_read", purpose="读取导航树与结构覆盖信息", result_policy="summary"),
        ToolSpec(name="insight_read", purpose="读取 insight 宿主容器中的候选记录", result_policy="excerpt"),
        ToolSpec(name="code_search", purpose="搜索定义、入口、主链候选", result_policy="summary"),
        ToolSpec(name="code_read", purpose="读取精确证据片段", result_policy="excerpt"),
        ToolSpec(name="deliver", purpose="提交最终结果", result_policy="summary"),
    ]


class ImpactMode:
    name = ModeName.IMPACT
    budget = BudgetPolicy(
        max_tool_calls=14,
        max_reads=10,
        max_escalations=3,
        max_result_chars=10000,
    )
    contract = PromptContract(
        identity="你是 PCE v2 的 impact 检索执行器。",
        objective="围绕已知目标给出直接影响点、主要传播链与修改风险。",
        stop_when=[
            "已经确认直接调用点或直接消费者",
            "已经得到足够支持风险判断的主要传播链",
        ],
        output_sections=["直接调用点", "数据契约/返回值消费者", "间接传播链", "风险", "建议修改顺序"],
    )
    tool_policy = ToolAssemblyPolicy(
        mode="impact",
        allowed_tools=["navigation_read", "insight_read", "code_search", "code_read", "impact_graph", "deliver"],
        denied_path_prefixes=[".pce/annotations"],
    )
    tool_specs = [
        ToolSpec(name="navigation_read", purpose="读取目标所属模块与覆盖结构", result_policy="summary"),
        ToolSpec(name="insight_read", purpose="读取相关历史认知条目", result_policy="excerpt"),
        ToolSpec(name="code_search", purpose="搜索直接引用与消费者候选", result_policy="summary"),
        ToolSpec(name="code_read", purpose="读取关键证据片段", result_policy="excerpt"),
        ToolSpec(name="impact_graph", purpose="读取调用链或引用图摘要", result_policy="summary"),
        ToolSpec(name="deliver", purpose="提交最终结果", result_policy="summary"),
    ]


class ReconcileMode:
    name = ModeName.RECONCILE
    budget = BudgetPolicy(
        max_tool_calls=0,
        max_reads=0,
        max_escalations=0,
        max_result_chars=0,
    )
    contract = PromptContract(
        identity="你是 PCE v2 的 reconcile 协调器。",
        objective="根据 dirty files 决定是否需要重建导航树，并清理过期 insight。",
        stop_when=["完成结构决策并输出结果"],
        output_sections=["decision", "changed_files", "removed_insights"],
    )
