from __future__ import annotations

from ..contracts import AssembledPrompt, PreparedRetrievalRequest


class PromptAssembler:
    """v2 prompt 装配器。

    第一版不追求复杂模板系统，只把已经在设计文档中确定的层级显式化：
    Identity / Mode / Policy / Context / Contract。
    """

    def assemble(self, request: PreparedRetrievalRequest) -> AssembledPrompt:
        policy_lines = [f"- {key}: {value}" for key, value in sorted(request.policy_context.items())]
        context_blocks = [
            self._render_block("Session Context", request.session_stable_context),
            self._render_block("Turn Context", request.turn_local_context),
            self._render_block("Policy", request.policy_context),
        ]
        system = "\n\n".join(
            [
                request.contract.identity,
                f"当前模式: {request.mode.value}",
                f"任务目标: {request.contract.objective}",
                "停止条件:\n" + "\n".join(f"- {item}" for item in request.contract.stop_when),
                "输出结构:\n" + "\n".join(f"- {item}" for item in request.contract.output_sections),
                "预算约束:\n"
                f"- max_tool_calls: {request.budget.max_tool_calls}\n"
                f"- max_reads: {request.budget.max_reads}\n"
                f"- max_escalations: {request.budget.max_escalations}\n"
                f"- max_result_chars: {request.budget.max_result_chars}",
                "策略提示:\n" + "\n".join(policy_lines),
            ]
        )
        return AssembledPrompt(system=system, context_blocks=context_blocks)

    def _render_block(self, title: str, payload: dict[str, str]) -> str:
        lines = [f"## {title}"]
        for key, value in payload.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
