# PCE — Precision Context Engine

**为 AI 编程 Agent 设计的代码库理解加速层。**

大型代码库理解的最大成本不是"搜索"，而是反复 grep、手工翻目录、拼接调用链带来的上下文消耗。PCE 通过 MCP 暴露的工具接口把入口定位、链路梳理、影响面分析从主 Agent 中拆出来：

- **节省主 Agent 上下文**：代码库调研由 PCE 独立完成，主 Agent 的上下文窗口不被探索过程消耗，得以在一轮会话中连续完成目标任务
- **降低模型调用成本**：PCE 的分析任务不依赖旗舰模型——推荐使用参数量小、速度快的轻量模型，性价比更高，响应更快

> **推荐模型**（实测性价比均衡）：`xiaomi/mimo-v2-flash` · `openai/gpt-5.4-mini` · `openai/gpt-5.4-nano`

## 工具一览

| 工具         | 用途                                         |
| ------------ | -------------------------------------------- |
| `pce_init`   | 绑定项目，构建索引与导航上下文               |
| `pce_query`  | 定位入口、梳理模块职责与主干调用链           |
| `pce_impact` | 分析已知目标的影响边界、下游传播链与变更风险 |
| `pce_sync`   | 代码变更后增量同步索引与认知状态             |
| `pce_status` | 查看初始化状态、索引信息与 staging 状态      |

典型工作流：

```
pce_init → pce_query → pce_impact → [修改代码] → pce_sync
```

- **目标未知** → 先用 `pce_query`
- **目标已知，评估波及面** → 用 `pce_impact`
- **改完代码后** → 用 `pce_sync`

---

## 快速接入（MCP）

推荐直接以 MCP 方式接入，无需手动启动服务。

### Claude Code

在 MCP 配置文件中添加：

```json
{
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
      "PCE_MODEL": "openai/gpt-5.4-nano",
      "PCE_API_KEY": "your_api_key",
      "PCE_TEMPERATURE": "1.0",
      "PCE_AGENT_TIMEOUT": "1200"
    }
  }
}
```

### Codex

```toml
[mcp_servers.pce]
command = "uvx"
startup_timeout_sec = 60
args = ["--python", "3.11", "--from", "git+https://github.com/Bluezeamer/PrecisionContextEngine", "pce", "serve"]
tool_timeout_sec = 1200

[mcp_servers.pce.env]
PCE_PROVIDER = "openrouter"
PCE_MODEL = "openai/gpt-5.4-nano"
PCE_API_BASE = "https://openrouter.ai/api/v1"
PCE_API_KEY = "your_api_key"
PCE_TEMPERATURE = "1.0"
PCE_AGENT_TIMEOUT = "1200"
```

---

## 环境变量

| 变量                               | 必填 | 说明                                                       |
| ---------------------------------- | ---- | ---------------------------------------------------------- |
| `PCE_PROVIDER`                     | 是   | LiteLLM provider，如 `openrouter` / `openai` / `anthropic` |
| `PCE_MODEL`                        | 是   | provider 下的模型名                                        |
| `PCE_API_KEY`                      | 是   | 对应模型的 API Key                                         |
| `PCE_API_BASE` / `PCE_BASE_URL`    | 否   | 自定义兼容端点                                             |
| `PCE_TEMPERATURE`                  | 否   | 全局温度，默认 `1.0`                                       |
| `PCE_AGENT_TIMEOUT`                | 否   | Agent 总超时（秒），默认 `600`                             |
| `PCE_COMPLETION_RETRIES_PER_MODEL` | 否   | 每模型 completion 重试次数，默认 `3`                       |
| `PCE_MODEL_FALLBACKS`              | 否   | fallback 模型链，逗号分隔                                  |

完整示例见 `.env.example`。

---

## 本地部署

**环境要求**：Python 3.11–3.12，`uv`

```bash
uv sync --all-extras
cp .env.example .env   # 按上表填写必填变量
uv run pce serve       # 以 stdio MCP server 方式运行
```

---

## License

GPL-3.0
