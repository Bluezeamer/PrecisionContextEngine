# PCE (Precision Context Engine)

> 有状态推理中间层，为代码理解提供上下文压缩和影响边界分析

## 项目定位

PCE 是一个代码库分析MCP工具，通过对serena的封装以及agent机制，实现基于语义的代码库分析和相关代码片段的精确定位和修改范围影响边界的分析，同时通过以下设计提升代码理解效率：

1. **上下文压缩** — 将代码库背景理解的 token 开销降低
2. **影响边界分析** — 在修改前给出完整的符号引用链，消除 build 试错循环
3. **知识积累** — 建立项目索引 + InsightCache 跨会话认知缓存，随使用加深对项目的理解

## 快速开始

### 环境要求

- [uv](https://docs.astral.sh/uv/) 包管理器（用于 `uvx` 命令）

### 安装

PCE 通过 `uvx` 按需运行，无需克隆仓库或手动安装依赖：

```bash
# 验证是否可运行（可选）
uvx --from git+https://github.com/Bluezeamer/PrecisionContextEngine pce serve

# 在目标项目目录下创建 .env 文件
cat > .env <<'EOF'
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_API_KEY=sk-or-...
EOF
```

> **说明**：Serena（PCE 的代码分析后端）和 PCE 本身均通过 `uvx` 自动获取与缓存，首次运行时会自动下载，后续复用缓存。
> 项目索引在 Agent 调用 `pce_init` 工具后自动构建，无需手动操作。
> 通过 Claude Code 使用时，MCP 配置会自动启动 PCE，通常无需手动在终端执行上面的命令。

### 配置 Claude Code

在 Claude Code 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "pce": {
      "command": "uvx",
      "args": ["--from", "git+<repository-url>", "pce", "serve"],
      "env": {
        "PCE_PROVIDER": "openrouter",
        "PCE_MODEL": "openai/gpt-4o-mini",
        "PCE_API_KEY": "sk-or-..."
      }
    }
  }
}
```

> 也可以不在 MCP config 中设置 `env`，将 `PCE_PROVIDER` / `PCE_MODEL` / `PCE_API_KEY` 等写入目标项目根目录的 `.env` 文件，PCE 在 `pce_init` 时自动加载（优先级低于 MCP env）。

### 在 Claude Code 中使用

每个会话开始时，**首先调用 `pce_init` 绑定目标项目**（仅需调用一次）：

```
> 使用 pce_init(project_path="/path/to/your/project") 初始化项目
```

初始化完成后，即可使用其余工具：

```
> 使用 pce_query 查询: "认证逻辑的入口在哪里?"
> 使用 pce_impact 分析修改影响: target="UserSession", change_type="modify"
> 使用 pce_status 查看当前索引状态和 warnings
```

> 若未调用 `pce_init`，`pce_query` / `pce_impact` / `pce_sync` 及写工具均不可用；`pce_status` 可在初始化前后随时调用。

## MCP 工具

### `pce_init`
绑定目标项目并触发初始化。**每个会话开始时首先调用一次**，传入目标项目的绝对路径。

- 内部流程：路径校验 → 状态机检查 → Serena 连接（`uvx` 拉起）→ `activate_project` 校验 → 全量/增量索引构建
- 同一路径重复调用可直接复用（幂等）；若已绑定其他路径则返回错误，需重启服务切换项目

### `pce_query`
自然语言查询代码库。PCE 内部通过 ReAct 循环驱动 Serena 工具检索代码结构，返回经过推理的**结构化 Markdown** 结果（默认包含结论、关键证据、相关符号、相关文件、不确定项）。

### `pce_impact`
变更影响分析。返回**结构化 Markdown** 结果（默认包含直接影响点、边界符号、风险、不确定项、建议修改顺序）。

### `pce_status`
查询 PCE 当前状态，包括索引信息、bootstrap 状态、warnings、InsightCache 统计。

### `pce_sync`
通知 PCE 代码库已发生大量变更。触发 Serena 重连 + 索引重建 + InsightCache 过期清理。适用于上层 Agent 完成一批修改后的批量沉淀。

> 日常小改动无需调用 `pce_sync` — PCE 的 FileWatcher 会实时追踪变更，查询时自动注入脏文件上下文。

### 写工具（透传）
PCE 还会透传 Serena 的符号编辑工具（加 `pce_` 前缀）：`pce_replace_symbol_body`、`pce_insert_after_symbol`、`pce_insert_before_symbol`、`pce_rename_symbol`。

## 技术架构

```
上层 Agent (Claude Code 等)
    ↓ MCP (pce_init / pce_query / pce_impact / pce_status / pce_sync)
PCE MCP Server
    ├─ PCEAgent (ReAct 循环，时间预算制)
    │   ├─ SubAgent spawn (深度限制 depth≤1)
    │   └─ InsightCache (跨会话认知缓存)
    ├─ Bootstrap（pce_init 驱动：路径校验 → Serena 连接 → activate_project → 索引构建）
    ├─ FileWatcher + StagingArea (实时文件变更追踪)
    └─ SerenaClient (MCP stdio 通信，uvx 按需拉起)
        ↓ MCP
Serena MCP Server (LSP + 文件系统)
```

### 核心模块

```
pce/
├── agent.py               ReAct Agent（模型降级路由 + SubAgent spawn）
├── agent_runtime/
│   ├── contracts.py       SpawnRequest/Result/schema/常量
│   └── spawner.py         invoke_spawn() 执行器
├── insight_cache.py       持久化认知缓存（跨会话知识积累）
├── serena_client.py       Serena MCP 客户端
├── tool_provider.py       ToolProvider Protocol
├── mock_tool_provider.py  离线 Mock（含 spawn_agent mock 规则）
├── models.py              Pydantic 数据模型
├── indexer.py             代码索引构建（全量/增量）
├── staging.py             文件变更暂存区 + FileWatcher
├── memory.py              Memory 读写管理
├── server.py              MCP Server 入口（bootstrap + 工具路由）
└── cli.py                 CLI 入口
```

## 环境变量

> 推荐通过 MCP config 的 `env` 段设置，避免明文写入磁盘文件。
> 也支持系统环境变量或项目根目录的 `.env` 文件（优先级：MCP env > 系统 env > `.env`）。

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `PCE_PROVIDER` | LiteLLM provider（**必填**，如 `openrouter` / `openai` / `anthropic`） | — |
| `PCE_MODEL` | 该 provider 下的模型名（**必填**） | — |
| `PCE_MODEL_FALLBACKS` | 同 provider 下的降级模型名列表（逗号分隔，可选） | 空（不降级） |
| `PCE_API_KEY` | litellm 通用 API Key（覆盖供应商默认 key） | — |
| `PCE_BASE_URL` | 自定义 Base URL（可选，常用于第三方中转或兼容端点） | — |
| `PCE_API_BASE` | `PCE_BASE_URL` 的兼容别名 | — |
| `PCE_SERENA_TIMEOUT` | Serena 连接/工具调用超时（秒） | `180` |
| `PCE_CONTEXT_WINDOW` | 上下文窗口大小（token） | `256000` |
| `PCE_LOG_LEVEL` | 日志级别 | `INFO` |

### 模型降级路由

当主模型遇到限流（429）、鉴权失败（401/403）或模型不存在（404）时，PCE 会自动切换到 `PCE_MODEL_FALLBACKS` 中的下一个候选模型。这里的 fallback 只写“模型名”部分，provider / base URL / API key 沿用主模型配置。litellm 自身已有单模型重试机制，PCE 只做模型级切换。

```bash
# OpenRouter 示例：在同一 provider=openrouter 下切换模型
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_MODEL_FALLBACKS=anthropic/claude-3.5-haiku,google/gemini-2.0-flash-001
```

### 供应商配置示例

PCE 底层使用 litellm，支持所有主流供应商。推荐在 MCP config 的 `env` 段配置：

```json
{
  "mcpServers": {
    "pce": {
      "command": "uvx",
      "args": ["--from", "git+<repository-url>", "pce", "serve"],
      "env": {
        "PCE_PROVIDER": "openai",
        "PCE_MODEL": "gpt-4o-mini",
        "PCE_API_KEY": "sk-...",
        "PCE_BASE_URL": "https://api.openai.com/v1"
      }
    }
  }
}
```

常用供应商写法：

```bash
# OpenRouter
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_API_KEY=sk-or-...

# OpenAI 直连
PCE_PROVIDER=openai
PCE_MODEL=gpt-4o-mini
PCE_API_KEY=sk-...

# Anthropic Claude
PCE_PROVIDER=anthropic
PCE_MODEL=claude-3-haiku-20240307
PCE_API_KEY=sk-ant-...

# OpenAI 兼容自建端点（如 vLLM / LocalAI）
PCE_PROVIDER=openai
PCE_MODEL=your-model-name
PCE_API_KEY=your-key
PCE_BASE_URL=http://localhost:8000/v1

# OpenRouter 下选择其他上游模型（模型名本身可带斜杠）
PCE_PROVIDER=openrouter
PCE_MODEL=anthropic/claude-sonnet-4
PCE_API_KEY=sk-or-...

# Anthropic 协议兼容中转
PCE_PROVIDER=anthropic
PCE_MODEL=claude-3-7-sonnet-latest
PCE_API_KEY=your-key
PCE_BASE_URL=https://your-anthropic-proxy.example.com
```

## 开发

### 安装开发依赖

```bash
uv sync --all-extras
```

### 代码风格

```bash
uv run black pce
uv run ruff check pce
uv run mypy pce
```

### 运行测试

```bash
# 单元测试（43 个场景，含 ReAct 循环健壮性 + spawn 路径 + fallback markers）
uv run python scripts/test_react_robustness.py

# 端到端集成测试（需要真实 Serena + LLM API Key）
uv run python scripts/test_e2e.py
```

## 文档

- [设计文档](docs/PCE_design.md)
- [SubAgent 架构设计](docs/design_subagent_architecture.md)
- [实施计划](docs/implementation_plan.md)
- [开发进度](docs/progress.md)

## License

MIT
