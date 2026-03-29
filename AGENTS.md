# PCE Tool Selection Rules

When MCP PCE tools are available in your toolset, follow these rules:

1. **Unknown location or macro understanding → `pce_query` first.** If you do not know which file contains the target, or the question is about what the project does, how a subsystem is organized, which modules are involved, or where a workflow starts, use `pce_query` before any directory traversal, glob, grep, or batch file reads.
2. **Known target, unknown impact → `pce_impact`.** If you know the symbol/file to change but not what it affects, use `pce_impact`.
3. **Use PCE before manual repo survey.** Do not start broad manual investigation of the repository for high-level understanding when PCE is available; first ask `pce_query` to narrow the space.
4. **Known file + known location → use Read/Grep/Edit directly.** PCE is not needed when you already have the precise path and position and only need local inspection or exact matching.

## Init / Topology / Digest Implementation Rules

5. 对 `init`、`pceignore`、`topology`、`digest` 这类阶段性流水线问题，优先重构阶段边界、facts 结构、工具预算与 Python 校验职责，不要通过局部补丁叠加修复症状。
6. 这类流水线实现以可用性、直观性、可收口为第一目标，不以兼容旧逻辑为目标。
7. 轻量特化 agent 必须显式受控：先给结构化 facts，再给小预算工具探索，再由 Python 侧做确定性校验与重试控制。
