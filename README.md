# PCE (Precision Context Engine)

> 有状态推理中间层,为代码理解提供上下文压缩和影响边界分析

## 项目定位

PCE 是一个运行在 Claude Code 与 Serena 之间的 **MCP Server**,通过以下能力提升代码理解效率:

1. **上下文压缩** - 将代码库背景理解的 token 开销从 50-60% 压缩到 10-15%
2. **影响边界分析** - 在修改前给出完整的符号引用链,消除 build 试错循环
3. **知识积累** - 建立项目索引,跨会话复用,随使用时间加深对项目的理解

## 快速开始

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- [git](https://git-scm.com/)（用于自动安装 Serena）

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd PrecisionContextEngine

# 安装依赖
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env,填入 OPENROUTER_API_KEY(或其他供应商的 Key)
# 若 Serena 启动较慢,可设置 PCE_SERENA_TIMEOUT(秒)
# 若 PyPI 访问受限,可设置 UV_DEFAULT_INDEX(镜像) 并按需配置 NO_PROXY/HTTP(S)_PROXY
```

> **说明**: Serena 无需手动安装。首次运行 `pce serve` 时会自动 clone 到 `serena/` 目录。

### 使用

#### 1. 初始化项目索引

```bash
uv run pce init --project /path/to/your/project
```

#### 2. 配置 Claude Code

在 Claude Code 的 MCP 配置中添加:

```json
{
  "mcpServers": {
    "pce": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/PrecisionContextEngine", "pce", "serve"],
      "env": {
        "PCE_PROJECT_PATH": "/path/to/your/project"
      }
    }
  }
}
```

> 注意: API Key 从 `.env` 文件自动加载,无需在 MCP 配置中重复填写。

#### 3. 在 Claude Code 中使用

```
> 使用 pce_query 查询: "认证逻辑的入口在哪里?"
> 使用 pce_impact 分析修改影响: target="UserSession", change_type="modify"
```

## 核心工具

PCE 提供 4 个 MCP 工具:

### `pce_init`
初始化项目索引,建立结构索引和引用索引

### `pce_query`
自然语言查询代码库,返回经过推理的结构化答案

### `pce_impact`
影响边界分析,给出完整的引用链和建议修改顺序

### `pce_status`
查询 PCE 当前状态和索引信息

## 技术架构

```
Claude Code (上层 Agent)
    ↓ MCP
PCE MCP Server
    ├─ Agent Loop (ReAct)
    ├─ Memory 管理
    └─ Serena Client
        ↓ MCP
Serena MCP Server (LSP + 文件系统)
```

**技术栈**:
- Python 3.11 + uv
- litellm (多供应商 LLM 调用)
- MCP 官方 Python SDK
- Pydantic (数据模型)

## 模型配置

PCE 通过 [litellm](https://docs.litellm.ai/) 支持多种 LLM 供应商。模型名称格式遵循 litellm 约定: `<provider_prefix>/<model_name>`。

### 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `PCE_MODEL` | Agent 推理使用的模型 | `openrouter/stepfun/step-3.5-flash:free` |
| `PCE_ANNOTATION_MODEL` | 索引构建时生成语义注解的模型 | `openrouter/stepfun/step-3.5-flash:free` |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | — |

### 供应商切换示例

```bash
# OpenRouter (默认,免费额度)
export OPENROUTER_API_KEY="sk-or-..."
# 模型名无需额外配置,默认值已适配 OpenRouter

# StepFun 直连
export STEPFUN_API_KEY="your-key"
export PCE_MODEL="step-3.5-flash"
export PCE_ANNOTATION_MODEL="step-3.5-flash"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-key"
export PCE_MODEL="anthropic/claude-3-haiku-20240307"
export PCE_ANNOTATION_MODEL="anthropic/claude-3-haiku-20240307"

# OpenAI
export OPENAI_API_KEY="your-key"
export PCE_MODEL="gpt-4o-mini"
export PCE_ANNOTATION_MODEL="gpt-4o-mini"
```

**核心模块**:
- `models.py` - 数据模型定义
- `memory.py` - Memory 读写管理
- `serena_client.py` - Serena MCP 客户端
- `indexer.py` - 索引构建
- `agent.py` - ReAct Agent Loop
- `server.py` - MCP Server 入口

## 开发

### 安装开发依赖

```bash
uv sync --all-extras
```

### 代码风格

```bash
# 格式化
uv run black pce tests

# Lint
uv run ruff check pce tests

# 类型检查
uv run mypy pce
```

### 运行测试

```bash
uv run pytest
```

## 文档

- [设计文档](docs/PCE_design.md)
- [实施计划](docs/implementation_plan.md)

## MVP 交付标准

1. `pce_init` 在 60 秒内完成中型项目索引
2. `pce_query` 返回正确定位,上下文消耗 < 3k tokens
3. `pce_impact` 返回完整引用链,无遗漏
4. 上层 Agent 无需执行 `ls`/`find`/`cat` 命令
5. 整体上下文开销降低 > 40%

## License

MIT

## 贡献

欢迎提交 Issue 和 Pull Request!
