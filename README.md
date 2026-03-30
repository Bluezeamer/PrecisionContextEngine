# PCE — Precision Context Engine

一个为 AI 编程 Agent 设计的有状态代码理解中间件。

PCE 位于你的 Agent（Claude Code、Codex 等）与代码库之间，负责处理入口定位、调用链追踪和变更影响分析这些繁琐工作——让 Agent 把上下文窗口留给真正的推理和编码，而不是反复搜索。

底层基于 [Serena](https://github.com/rosalab/serena) 的结构化分析能力，上层增加了 ReAct Agent 循环、增量索引和模块级认知缓存。

## 它做什么

PCE 通过 MCP 协议暴露一组工具：

| 工具         | 用途                                           |
| ------------ | ---------------------------------------------- |
| `pce_init`   | 绑定目标项目，启动 Serena，构建代码索引        |
| `pce_query`  | 定位入口、追踪调用链、厘清模块职责             |
| `pce_impact` | 对已知变更目标分析影响边界（签名、字段、接口） |
| `pce_sync`   | 代码修改后重建索引                             |
| `pce_status` | 查看初始化状态、索引统计、告警信息             |

典型工作流：

```
pce_init(project_path=...)      # 绑定项目
  → pce_query(...)              # 我要找的东西在哪？
  → pce_impact(...)             # 改了它会影响什么？
  → Agent 修改代码
  → pce_sync()                  # 同步索引
```

目标还不明确时用 `pce_query`；目标已知、需要了解波及面时用 `pce_impact`。

## 快速开始

### 安装依赖

```bash
uv sync --all-extras
```

### 配置环境

```bash
cp .env.example .env
```

最小配置：

```env
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_API_KEY=sk-or-...
```

PCE 使用 [LiteLLM](https://github.com/BerriAI/litellm) 作为 LLM 调用层，支持 OpenAI、Anthropic、OpenRouter、本地端点等所有 LiteLLM 兼容的 provider。

### 环境变量参考

#### LLM 配置（核心）

| 变量                  |  必填  | 默认值 | 说明                                                                         |
| --------------------- | :----: | ------ | ---------------------------------------------------------------------------- |
| `PCE_PROVIDER`        | **是** | —      | LiteLLM provider 名称，如 `openrouter`、`openai`、`anthropic`                |
| `PCE_MODEL`           | **是** | —      | 该 provider 下的模型名。OpenRouter 下模型名可含斜杠，如 `openai/gpt-4o-mini` |
| `PCE_API_KEY`         |   是   | —      | 统一传给 LiteLLM 的 API Key                                                  |
| `PCE_BASE_URL`        |   否   | —      | 自定义 API 端点，用于第三方中转或本地部署（如 vLLM、LocalAI）                |
| `PCE_API_BASE`        |   否   | —      | `PCE_BASE_URL` 的兼容别名，效果相同                                          |
| `PCE_MODEL_FALLBACKS` |   否   | —      | 同 provider 下的模型降级链，逗号分隔，如 `gpt-4o-mini,gpt-4.1-mini`          |
| `PCE_TEMPERATURE`     |   否   | —      | 全局模型温度总控；一旦设置，会覆盖所有更细粒度的 temperature 配置             |
| `PCE_AGENT_TEMPERATURE` | 否   | —      | 主 `query` / `impact` / ReAct 主链温度                                       |
| `PCE_TOPOLOGY_TEMPERATURE` | 否 | —      | `pceignore` / `navigation_tree` / topology 增量链温度                        |
| `PCE_DIGEST_TEMPERATURE` | 否  | —      | digest stage1 / stage2 / stageB / stageC 温度                                |
| `PCE_ANNOTATION_TEMPERATURE` | 否 | —    | `annotation_writer` 独立补全链温度                                           |

#### 运行配置（可选）

| 变量                 | 默认值       | 说明                                                                           |
| -------------------- | ------------ | ------------------------------------------------------------------------------ |
| `PCE_PROJECT_PATH`   | 当前工作目录 | 目标项目根路径（绝对路径）。MCP 模式下通常不需要设置，通过 `pce_init` 传入即可 |
| `PCE_CONTEXT_WINDOW` | `256000`     | Agent 上下文窗口大小（token）。影响动态注入块的软上限（= 窗口 / 10）           |
| `PCE_SERENA_TIMEOUT` | `180`        | Serena MCP 启动 / 初始化超时，单位秒                                           |
| `PCE_LOG_LEVEL`      | `INFO`       | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR`                               |
| `PCE_TRACE_DIR`      | —            | 设置后启用 JSONL Trace 输出，将 Agent 推理过程写入该目录                       |

#### 网络与环境（可选）

| 变量               | 说明                                                                               |
| ------------------ | ---------------------------------------------------------------------------------- |
| `UV_DEFAULT_INDEX` | uv 依赖下载源，网络受限时可切换镜像，如 `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `NO_PROXY`         | 代理绕过列表                                                                       |

#### 常见接入方式示例

```bash
# OpenRouter
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_API_KEY=sk-or-...

# OpenAI 直连
PCE_PROVIDER=openai
PCE_MODEL=gpt-4o-mini
PCE_API_KEY=sk-...

# Anthropic 直连
PCE_PROVIDER=anthropic
PCE_MODEL=claude-3-haiku-20240307
PCE_API_KEY=sk-ant-...

# OpenAI 兼容端点（vLLM / LocalAI / 第三方中转）
PCE_PROVIDER=openai
PCE_MODEL=your-model-name
PCE_API_KEY=your-key
PCE_BASE_URL=http://localhost:8000/v1

# Anthropic 协议兼容中转
PCE_PROVIDER=anthropic
PCE_MODEL=claude-3-7-sonnet-latest
PCE_API_KEY=your-key
PCE_BASE_URL=https://your-anthropic-proxy.example.com
```

完整示例见 [.env.example](./.env.example)。

### 启动

```bash
uv run pce serve
```

PCE 以 stdio MCP server 方式运行，目标项目通过 `pce_init` 在运行时绑定。

## MCP 接入

在你的 Agent MCP 配置中添加（以 Claude Code 为例）：

```json
{
  "mcpServers": {
    "pce": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Bluezeamer/PrecisionContextEngine",
        "pce",
        "serve"
      ],
      "env": {
        "PCE_PROVIDER": "openrouter",
        "PCE_MODEL": "openai/gpt-4o-mini",
        "PCE_API_KEY": "sk-or-..."
      }
    }
  }
}
```

## 设计思路

- **双工具分治。** `query` 负责定位，`impact` 负责边界分析。拆开而非合成一个大而全的黑箱，让每个工具的契约可测试、可预期。
- **Markdown 优先输出。** 结果是结构化 Markdown——人可以直接阅读，Agent 可以直接引用或追问，不依赖脆弱的 JSON schema。
- **增量认知。** 模块级 digest、baseline、annotation 在调用间持久化，Agent 不需要每次会话都从零重新理解项目。
- **Ignore-first 文件发现。** 先尊重项目 `.gitignore`，再叠加 PCE 自身排除规则。不会意外索引 `node_modules` 或 `.venv`。

## 项目结构

```
pce/
├── agent.py                 ReAct Agent（query & impact 循环）
├── server.py                MCP 工具注册与路由
├── indexer.py               全量 / 增量索引构建
├── digest_agent.py          模块级 digest 生成
├── digest_delta_builder.py  增量 digest 更新
├── insight_cache.py         跨调用认知缓存
├── module_registry.py       模块 identity 追踪
├── serena_client.py         Serena MCP 客户端适配
├── memory.py                .pce/ 目录布局与持久化
├── staging.py               脏文件追踪 & 文件监听
└── file_discovery.py        Ignore-aware 文件发现
```

## 开发与测试

```bash
# 格式检查
uv run --with black python -m black --check pce scripts

# 鲁棒性测试
uv run python scripts/test_react_robustness.py

# impact 契约测试
uv run python scripts/test_impact_contract.py
```

## 当前状态

PCE 处于早期活跃开发阶段。`query` 已经可以稳定用于日常代码导航；`impact` 能提供有价值的影响边界参考，但长尾边界仍在持续收敛。

现阶段它最适合作为编程 Agent 的理解加速层——而不是一个保证穷尽的静态分析器。

## 环境要求

- Python 3.11 – 3.12
- [uv](https://github.com/astral-sh/uv) 包管理工具

## 许可证

[GPL-3.0](./LICENSE)

---

<details>
<summary>English</summary>

# PCE — Precision Context Engine

A stateful code-understanding middleware for AI coding agents.

PCE sits between your agent (Claude Code, Codex, etc.) and the codebase. It handles entry-point location, call-chain tracing, and change-impact analysis — so the agent can spend its context window on reasoning and coding rather than repeated searches.

Built on [Serena](https://github.com/rosalab/serena) for structural analysis, with an added ReAct agent layer, incremental indexing, and module-level cognitive caching.

## What it does

PCE exposes a small set of MCP tools:

| Tool         | Purpose                                                                         |
| ------------ | ------------------------------------------------------------------------------- |
| `pce_init`   | Bind to a project, start Serena, build the code index                           |
| `pce_query`  | Locate entry points, trace call chains, clarify module responsibilities         |
| `pce_impact` | Analyze the blast radius of a known change target (signature, field, interface) |
| `pce_sync`   | Re-index after code modifications                                               |
| `pce_status` | Check initialization state, index stats, warnings                               |

Typical workflow:

```
pce_init(project_path=...)      # bind to the project
  → pce_query(...)              # where is the thing I need?
  → pce_impact(...)             # what breaks if I change it?
  → agent edits code
  → pce_sync()                  # update the index
```

Use `pce_query` when you don't yet know where to look. Use `pce_impact` when the target is clear and you need its dependency surface.

## Getting started

### Install

```bash
uv sync --all-extras
```

### Configure

```bash
cp .env.example .env
```

Minimal setup:

```env
PCE_PROVIDER=openrouter
PCE_MODEL=openai/gpt-4o-mini
PCE_API_KEY=sk-or-...
```

PCE uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood — any provider it supports (OpenAI, Anthropic, OpenRouter, local endpoints, etc.) works out of the box.

### Environment variables

#### LLM configuration (required)

| Variable              |  Required   | Default | Description                                                                                  |
| --------------------- | :---------: | ------- | -------------------------------------------------------------------------------------------- |
| `PCE_PROVIDER`        |   **Yes**   | —       | LiteLLM provider name, e.g. `openrouter`, `openai`, `anthropic`                              |
| `PCE_MODEL`           |   **Yes**   | —       | Model name under the provider. May contain slashes for OpenRouter, e.g. `openai/gpt-4o-mini` |
| `PCE_API_KEY`         | Recommended | —       | API key passed to LiteLLM                                                                    |
| `PCE_BASE_URL`        |     No      | —       | Custom API endpoint for proxies or local deployments (vLLM, LocalAI, etc.)                   |
| `PCE_API_BASE`        |     No      | —       | Legacy alias for `PCE_BASE_URL`                                                              |
| `PCE_MODEL_FALLBACKS` |     No      | —       | Comma-separated fallback models under the same provider                                      |
| `PCE_TEMPERATURE`     |     No      | —       | Global master temperature; when set, overrides all finer-grained temperature settings        |
| `PCE_AGENT_TEMPERATURE` |   No      | —       | Temperature for the main `query` / `impact` / ReAct chain                                    |
| `PCE_TOPOLOGY_TEMPERATURE` | No     | —       | Temperature for `pceignore` / topology / navigation chains                                   |
| `PCE_DIGEST_TEMPERATURE` | No       | —       | Temperature for digest stage1 / stage2 / stageB / stageC                                     |
| `PCE_ANNOTATION_TEMPERATURE` | No   | —       | Temperature for the standalone `annotation_writer` completion path                           |

#### Runtime configuration (optional)

| Variable             | Default  | Description                                                                                  |
| -------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `PCE_PROJECT_PATH`   | cwd      | Target project root (absolute path). Usually not needed in MCP mode — use `pce_init` instead |
| `PCE_CONTEXT_WINDOW` | `256000` | Agent context window size in tokens. Dynamic injection soft limit = window / 10              |
| `PCE_SERENA_TIMEOUT` | `180`    | Serena MCP startup timeout in seconds                                                        |
| `PCE_LOG_LEVEL`      | `INFO`   | Log level: `DEBUG` / `INFO` / `WARNING` / `ERROR`                                            |
| `PCE_TRACE_DIR`      | —        | When set, writes JSONL traces of agent reasoning to this directory                           |

See [.env.example](./.env.example) for full examples including various provider setups.

### Run

```bash
uv run pce serve
```

PCE runs as a stdio-based MCP server. The target project is bound at runtime via `pce_init`.

## MCP integration

Add to your agent's MCP config (e.g. Claude Code `settings.json`):

```json
{
  "mcpServers": {
    "pce": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Bluezeamer/PrecisionContextEngine",
        "pce",
        "serve"
      ],
      "env": {
        "PCE_PROVIDER": "openrouter",
        "PCE_MODEL": "openai/gpt-4o-mini",
        "PCE_API_KEY": "sk-or-..."
      }
    }
  }
}
```

## Design choices

- **Two-tool split.** `query` locates, `impact` analyzes boundaries. Separate contracts instead of a monolithic black box.
- **Markdown-first output.** Structured Markdown — human-readable, agent-quotable, no fragile JSON schemas.
- **Incremental cognition.** Module-level digests, baselines, and annotations persist across calls.
- **Ignore-first file discovery.** Respects `.gitignore` before applying PCE's own exclusions.

## Requirements

- Python 3.11 – 3.12
- [uv](https://github.com/astral-sh/uv)

## License

[GPL-3.0](./LICENSE)

</details>
