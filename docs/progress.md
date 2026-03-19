# PCE 开发进度

> 最后更新：2026-03-19
> 设计文档：[PCE_design.md](PCE_design.md) · [design_subagent_architecture.md](design_subagent_architecture.md) · [implementation_plan.md](implementation_plan.md)

---

## 维护约定

**本文件是项目进度的唯一档案。**
- 每次会话结束前，将本轮完成内容、关键决策、下一步写入 Changelog
- Serena memory 仅作临时草稿，以本文件为准
- 文件路径：`docs/progress.md`（Git 托管，天然版本化）

---

## 项目概况

PrecisionContextEngine (PCE) 是一个以 MCP 服务形式对外暴露的代码理解引擎。
上层 Agent（如 Claude Code）通过以下工具调用 PCE：

- `pce_init`：初始化并绑定目标项目（会话开始时调用一次）
- `pce_query`：自然语言代码理解
- `pce_impact`：变更影响分析
- `pce_status` / `pce_sync`：状态与索引刷新

内部通过 LLM ReAct 循环驱动 Serena 代码分析工具完成调研，结果以结构化形式返回。

## 技术栈

- Python 3.12+，依赖管理：`uv`
- LLM 调用：OpenRouter API（兼容 OpenAI 格式）
- 代码分析：Serena MCP（通过 `pce/serena_client.py` 封装，via `uvx`）
- 数据模型：Pydantic v2
- 并发：asyncio

---

## 已完成

### 基础层

- `pce/models.py` — 完整数据模型层
  - 核心模型：ProjectMeta, FileMeta, SymbolRef, ReferenceEdge, IndexEntry, IndexSnapshot
  - 会话模型：MemoryItem, SessionState（死代码，待清理）
  - MCP 协议：QueryRequest/Response, ImpactRequest/Response, StatusResponse（含 InsightStats 字段）
  - **改造**：QueryRequest/Response/ImpactRequest/Response 移除 `session_id` 字段
  - **改造**：ImpactResponse.impact_chain/boundary 和 QueryResponse.related_symbols 改为 `list[dict[str, Any]]`（轻量 dict，不再要求 UUID）
  - Insight 模型：InsightConfidence, InsightEntry, InsightIndexRecord, InsightIndex, InsightStats
  - **新增**：`InitResponse`（pce_init 响应，含 status/init_mode/file_count/warnings/error）
- `pce/indexer.py` — 代码索引构建
- `pce/staging.py` — 文件变更追踪（dirty_files.json）
- `pce/memory.py` — 会话记忆管理
- `pce/serena_client.py` — Serena MCP 客户端封装
  - **改造**：`connect()` 去掉 `serena_install_path` 参数，改用 `uvx --from git+https://github.com/oraios/serena`
  - **改造**：`DEFAULT_TIMEOUT_SECONDS` 从 30 提升至 180，可通过 `PCE_SERENA_TIMEOUT` 环境变量配置
- `pce/tool_provider.py` — ToolProvider Protocol（支持 duck typing）

### Agent 层

- `pce/agent.py` — PCEAgent 核心（**无状态**，时间驱动 ReAct 循环）
  - **改造**：完全无状态化 — 移除 `self._sessions`，每次 `query()`/`impact()` 从 Memory 快照重新起步
  - **改造**：per-request 日志追踪 — 8 字符 hex `req_id`（ContextVar 传播），所有工具调用（Serena/spawn/acknowledge）带 `[req=xxx] round=N` INFO 日志
  - **新增**：`_TraceWriter` — 可选 JSONL trace 文件（`PCE_TRACE_DIR` 环境变量控制），LRU 保留 50 个文件
  - **新增**：`_key_arg_preview()` — 工具调用日志预览（`tool(key='val')` 格式），`_PRIMARY_ARG` 映射表
  - **新增**：`_req_id_var: ContextVar[str]` — 通过 asyncio context 自动传播给 spawn 子 Agent
  - `_run_react_loop`：新增 `trace` 参数、`round_num` 计数，所有消息路径均有 trace 写入
  - `_invoke_tool`：新增计时，返回 `_elapsed_seconds`/`_preview` 内部字段（由 `_run_react_loop` pop 后用于日志）
  - `_completion`：模型降级路由（fallback chain），主模型限流/鉴权失败时自动切换候选模型
  - `_maybe_compact`：上下文压缩（超 token 阈值时），LLMCompletionError 穿透不吞
  - `query()` / `impact()`：两个主入口，无 `session_id` 参数，deliver 后自动调用 `_persist_insights`
  - `_build_system_prompt`：每次调用注入 InsightCache top-k 动态认知块
  - `_persist_insights` + 辅助方法：deliver 结果蒸馏写入 InsightCache
  - **改造**：SYSTEM_PROMPT_HEADER 增加精确 JSON schema + in-context learning 示例（deliver answer 必须是严格 JSON）
  - **新增**：`_extract_lightweight_symbol()` / `_extract_lightweight_edge()` — 从 LLM 输出提取轻量结构化字段（不要求 UUID）
  - **改造**：`_parse_impact_response` / `_parse_query_response` 改用轻量提取，不再 `model_validate` 强校验
  - **改造**：`impact()` user prompt 替换为完整字段 schema + 字段说明
  - SYSTEM_PROMPT_HEADER：含 spawn_agent 使用策略章节（判断标准、推荐/不推荐场景、结果处理）
  - `LLMCompletionError`：结构化异常，携带完整降级记录（尝试过的模型、错误类型、原因）
  - `_should_fallback_model`：基于 `isinstance` 的异常分类，判断是否触发降级
  - `__REACT_LLM_EXHAUSTED__`：新增终止 marker，模型降级链耗尽时返回
- `pce/mock_tool_provider.py` — MockToolProvider（离线测试）
  - 支持 replay（预录回放）+ rules（规则匹配）+ fallback
  - RecordingProxy：透明代理，录制真实交互
  - `_rule_spawn_agent`：spawn 场景 mock 规则，错误码引用 `SpawnErrorCode` 枚举（不硬编码字符串）

### SubAgent 架构

- `pce/agent_runtime/contracts.py`
  - SpawnRequest / SpawnResult dataclass（含 `to_tool_content()` 序列化方法）
  - SpawnStatus / SpawnErrorCode 枚举
  - SPAWN_AGENT_TOOL litellm 工具 schema
  - 常量：MIN_SPAWN_BUDGET=8s, MAX_SPAWN_BUDGET_RATIO=0.70, MAX_SPAWNS_PER_LOOP=3
- `pce/agent_runtime/spawner.py`
  - `invoke_spawn()`：失败不抛异常，统一返回 SpawnResult
  - 深度检查（parent_depth>=1 → SUBAGENT_DEPTH_EXCEEDED）
  - 预算：remaining < MIN_SPAWN_BUDGET 拒绝；否则只向下 clamp 不向上抬高
  - `_CHILD_SYSTEM_PROMPT`：5 条规则，强调精炼输出与禁止递归 spawn
- `pce/agent.py` spawn 集成
  - depth==0 才注入 SPAWN_AGENT_TOOL（递归保护）
  - 分拣阶段增加 spawn_calls 分支，结果序列化后回注 messages
  - spawn_count 限制（MAX_SPAWNS_PER_LOOP=3）
  - 子 Agent 成功结论同步写入 InsightCache

### 服务层

- `pce/server.py` — MCP 服务端，**一键安装改造完成**，**session_id 已移除**
  - **改造**：pce_query/pce_impact 工具 schema、handle_query/handle_impact 签名、call_tool 路由均移除 `session_id`
  - **改造**：pce_query/pce_impact 工具描述增加最佳实践指导和调用示例
  - **新增**：`_bootstrap` 中 `InsightCache.sweep_stale()` + `cleanup_stale()` — 初始化时清理过时 insight
  - **新增**：`handle_sync` 中 `sweep_stale()` 后补调 `cleanup_stale()` — sync 时标记+删除过时条目
  - **新增**：`_format_dirty_context` 加 `_DIRTY_INJECT_MAX_FILES=50` 截断上限 + 超限提示
  - `PCEContext` 改为空壳启动：`__init__()` 无 project_path，所有运行时组件为 `None`
  - 四态状态机：`uninitialized → initializing → initialized | failed`
  - **新增** `pce_init` 工具 + `handle_init()`：同步阻塞到完成，含并发保护、同路径重试、不同路径拒绝
  - **新增** `_require_initialized()`：统一守卫 pce_query/impact/edit/sync 入口
  - `_bootstrap(project_path, init_mode)` 重构：接受路径参数，内部创建 SerenaClient/InsightCache/PCEAgent，FileWatcher 在路径绑定后立即启动
  - `_build_tools()` 按初始化状态分路：未初始化只暴露 pce_init + pce_status；初始化后暴露全套
  - `handle_status()` 去掉 project_path 参数，未初始化时安全返回空值兜底
  - `serve()` 去掉 PCE_PROJECT_PATH、SERENA_PATH 环境变量读取，去掉 `_ensure_serena()`
  - finally 块各自独立 try，watcher.stop() 失败不影响 serena_client.disconnect()
- `pce/cli.py` — CLI 入口，去掉 `--project` 和 `--serena` 参数，serve 子命令零参数

### Insight Cache

- `pce/insight_cache.py` — 持久化认知缓存
  - 存储：`.pce/insights/index.json` + `.pce/insights/entries/{uuid}.json`
  - `upsert` / `get_top_k` / `sweep_stale` / `cleanup_stale` / `stats`
  - **改造**：`cleanup_stale` 增加孤儿 entry 文件清理（扫描 entries 目录删除不在 index 中的文件）
  - 并发安全：写操作持 asyncio.Lock，原子写用 os.replace

### 测试工具

- `scripts/agent_playground.py` — 离线测试 CLI（mock/repl 双模式，支持 InsightCache）
- `scripts/test_react_robustness.py` — ReAct 循环健壮性测试（**43/43 通过**）
  - T01~T15：基础路径（deliver/工具调用/纠正/截断/超时/步数）
  - T16~T20：spawn 路径（成功回注 / DEPTH_EXCEEDED / BUDGET_REJECTED / 次数上限 / 非法 JSON）
  - fallback marker 测试：含 `__REACT_LLM_EXHAUSTED__`
  - `run_loop` 新增 `depth/deadline/observe` 关键字参数（向后兼容）
- `scripts/test_e2e.py` — 端到端集成测试（真实 Serena + 真实 LLM，**2/2 通过**）
  - pce_query：自然语言代码理解（101.58s，含多轮 Serena 工具调用）
  - pce_impact：变更影响分析（79.3s，生成结构化风险/未知项列表）

---

## 待完成

### 清理项
- [ ] 删除死代码：`SessionState`（`pce/models.py:216`）+ `memory.py` 的 `load_session`/`save_session`/`clear_session`
- [ ] 删除死代码：`memory.py` 中 sessions 目录相关函数（`_session_path`/`_sessions_dir`）

### 质量项
- [x] ~~pce_impact 分析质量~~：已修复（轻量 dict 解析 + prompt schema + in-context learning）
- [x] ~~pce_impact 稳定性~~：transport 崩溃问题随无状态化修复消失
- [x] ~~pce_query 单轮稳定性~~：回归验证通过（2/2 query + 2/2 impact）
- [x] ~~`_bootstrap` insight 清理~~：已实现 sweep_stale + cleanup_stale
- [x] ~~`cleanup_stale` 孤儿清理~~：已增加 entries 目录扫描

### 架构演进（高优先级）
- [ ] annotations 按模块拆分存储（当前是单体 `annotations.md`，大项目会膨胀）
- [ ] `_build_system_prompt` 相关性过滤（根据 query 匹配 scope，按需加载模块 annotation）
- [ ] system prompt 总量上限检查 + 降级策略（超限时只注入 structure.md + 直接相关 annotation）
- [ ] insight 蒸馏去重（避免与 annotations 内容重复挤占 token 配额）
- [ ] dirty file 注入时引导 agent 用 `get_symbols_overview` 做轻量验证（而非全文读取）

### 可选方向
- [ ] CLI / playground 文档完善
- [ ] 性能调优（并发 spawn、InsightCache 读取优化）

---

## 架构速查

```
pce_init(project_path) — 会话开始时调用一次
  │
  ├─ 路径校验 + 状态机检查（并发/重试/路径冲突）
  ├─ StagingArea + FileWatcher 创建并启动（立即，不等索引完成）
  ├─ InsightCache.sweep_stale() + cleanup_stale()（清理过时+孤儿 insight）
  ├─ SerenaClient.connect()（uvx 拉起 Serena）
  ├─ activate_project 校验
  └─ _run_index_refresh()（全量或增量）→ 返回 InitResponse

pce_query 请求
  │
  ├─ _require_initialized() 守卫
  │
  ├─ PCEAgent.query()  — 无状态，每次重建 messages
  │    ├─ req_id 生成 + ContextVar 设置 + TraceWriter 创建
  │    ├─ _build_system_prompt()（含 InsightCache top-k 注入）
  │    └─ _run_react_loop(depth=0, deadline=now+max_seconds, trace=...)
  │         ├─ Serena 工具调用（asyncio.gather 并发）
  │         ├─ spawn_agent → invoke_spawn()
  │         │    └─ _run_react_loop(depth=1, deadline=child_deadline)
  │         │         ├─ Serena 工具调用
  │         │         └─ deliver → SpawnResult 回注父消息
  │         └─ deliver → 终止
  │
  ├─ 同步返回 QueryResponse
  │
  └─ _persist_insights()  →  InsightCache.upsert()（deliver 后异步写入）

pce_sync 请求
  │
  ├─ _require_initialized() 守卫
  ├─ _sync_lock 内：Serena 重连 → 增量/全量索引重建
  └─ 锁外：InsightCache.sweep_stale() + cleanup_stale()（标记+删除过时条目+孤儿清理）

PCEContext 状态机
  uninitialized
    └─ pce_init() → initializing
         ├─ 成功 → initialized
         └─ 失败 → failed（同路径可重试）

.pce/
├── meta.json
├── structure.md
├── annotations.md
├── references.json
├── dirty_files.json
└── insights/
    ├── index.json
    └── entries/{uuid}.json

pce/
├── agent.py               主 Agent（PCEAgent）
├── agent_runtime/
│   ├── contracts.py       SpawnRequest/Result/schema/常量
│   └── spawner.py         invoke_spawn() 执行器
├── insight_cache.py       持久化认知缓存
├── serena_client.py       Serena MCP 客户端（uvx 启动）
├── tool_provider.py       ToolProvider Protocol
├── mock_tool_provider.py  离线 Mock（含 spawn_agent mock 规则）
├── models.py              数据模型（含 InitResponse）
├── server.py              MCP 服务端（pce_init 驱动初始化）
└── cli.py                 CLI 入口（零参数 serve）
```

---

## 关键决策

| 决策 | 结论 | 原因 |
|------|------|------|
| Agent 框架 | 不引入第三方框架 | spawn 需要拦截 LLM tool_call 自行调度，框架反而是障碍 |
| SubAgent 模块位置 | `pce/agent_runtime/`（不用 `pce/agent/`） | 避免与 `pce/agent.py` 产生 Python 包命名冲突 |
| spawn 预算策略 | 只向下 clamp，不向上抬高 | 防止突破 MAX_SPAWN_BUDGET_RATIO 上限 |
| spawn 次数限制 | 循环局部计数（非跨 query） | 每次 query 独立限制 3 次 spawn 是合理粒度 |
| 进度档案位置 | `docs/progress.md`（唯一） | Git 托管，天然版本化；Serena memory 仅临时草稿 |
| Insight 存储结构 | index/entries 分离 | 读路径按需加载 content，写路径独立不重写全量 |
| stale 判断时机 | 懒校验（get_top_k）+ 批量（sweep_stale） | 兼顾准确性与性能 |
| Insight 注入时机 | 每次调用（`_build_system_prompt`） | 无状态化后每次重建 system prompt |
| sweep_stale 调用位置 | `_sync_lock` 外 | sweep 需遍历并哈希所有条目，放锁内会拉长临界区阻塞并发同步请求 |
| 模型降级策略 | fallback chain，不重试同一模型 | litellm 自身已有重试，上层只做模型级切换 |
| 异常分类方式 | isinstance(litellm_exc.*) | 比类名字符串比较更稳定，不受重命名影响 |
| litellm.Timeout 处理 | 转换为 asyncio.TimeoutError | 统一超时类型，由 _run_react_loop 的 timeout_retries 机制处理 |
| bootstrap 策略 | **改为 pce_init 驱动**（原为 eager 启动时） | 支持全局安装零配置；agent 在会话开始时主动传入项目路径 |
| pce_init 同步/异步 | 同步阻塞到完成 | 初始化是一次性的项目认知构建，时间成本合理；避免 agent 轮询 |
| Serena 启动方式 | uvx 按需下载（原为本地 clone） | 零本地依赖，一键安装；uvx 自动缓存，冷启动后续快速 |
| 项目路径传递 | pce_init 工具传入（原为 PCE_PROJECT_PATH 环境变量） | 支持全局安装、多项目共用同一 MCP server |
| FileWatcher 启动时机 | 路径校验通过即启动，不等索引完成 | 避免初始化期间（可能较长）遗漏文件变更 |
| _bootstrap_event.clear() 位置 | 仅在 handle_init 状态切换时（不在 _bootstrap 内） | 防止并发 init 竞态：第二个等待方可能在 clear 后错过 set |
| 失败后组件重置 | agent/insight_cache/serena_client 置 None | 避免半初始化对象在重试时残留 |
| Serena 激活校验 | bootstrap 中显式调用 activate_project | Serena 启动时激活失败会被静默吞掉，再调一次可捕获失败 |
| bootstrap 与 onboarding 分离 | bootstrap 不依赖 LLM | Serena onboarding 需要 LLM 写 memory，不应作为服务可用的前置条件 |
| PCEAgent 无状态化 | 移除 `_sessions`，每次请求重建 messages | 设计意图是每次从 Memory 快照起步；旧 session 复用导致消息序列非法 bug |
| 调试可观测性 | req_id INFO 日志 + 可选 JSONL trace | 替代 session 历史作为排障手段；trace 仅 PCE_TRACE_DIR 设置时生效，不增加默认开销 |
| req_id 不对外暴露 | PCE 内部消费，不返回给调用方 | PCE 作为工具层做好自身日志，不污染上层 Agent 接口 |
| impact 结构化输出 | 轻量 dict 替代 ReferenceEdge/SymbolRef | LLM 无法凭空生成 UUID，旧 model_validate 静默失败导致结构化字段永远为空 |
| deliver answer 格式 | 严格 JSON 字符串 + in-context learning 示例 | 消除 prompt 与 deliver 工具之间的语义断层 |
| Insight 生命周期闭环 | init 时 sweep+cleanup，sync 时 sweep+cleanup | 避免 insight 只增不减导致上下文膨胀 |
| dirty file 注入截断 | 50 文件上限 + 超限提示调 pce_sync | 防御性策略，避免极端情况撑爆 token |
| 认知架构演进方向 | annotations 按模块拆分 + 渐进式披露 | 治本：起始上下文默认轻量，按需加载局部认知 |

---

## Changelog

### 2026-03-19 14:08
本轮完成：pce_impact 结构化输出修复 + Insight 生命周期闭环 + 认知架构探索
主体更新："已完成"（Agent 层/基础层/服务层/InsightCache 改造）、"待完成"（质量项标记完成 + 新增架构演进方向）、"架构速查"（init/sync 流程更新）、"关键决策"（5 条新增）
下一步：annotations 按模块拆分 + `_build_system_prompt` 渐进式披露改造

### 2026-03-19 08:22
本轮完成：PCEAgent 无状态化 + per-request 日志追踪（`pce/agent.py`、`pce/models.py`、`pce/server.py`）
主体更新："已完成"（Agent 层/基础层/服务层改造描述）、"待完成"（新增清理项+质量项）、"关键决策"（3 条新增）、"架构速查"（pce_query 流程更新）
下一步：回归验证 pce_query 单轮稳定性 → 排查 pce_impact 分析质量与 transport 崩溃

### 2026-03-18（本轮）
本轮完成：API 供应商配置通用化改造
- 新增 `pce/_env.py`：`get_env_text()` + `get_completion_overrides()`，集中管理 `PCE_API_KEY`/`PCE_API_BASE` 读取
- `pce/agent.py`：去掉模块级 `MODEL`/`MODEL_FALLBACKS` 常量（导入冻结问题）；`__init__` 运行时读取 `PCE_MODEL`（必填，未配置抛 ValueError）；`_completion()` 加 `completion_overrides` 透传
- `pce/indexer.py`：去掉 `DEFAULT_MODEL` 常量；`_generate_annotations()` 运行时解析模型（PCE_ANNOTATION_MODEL → PCE_MODEL → 跳过）；加 `completion_overrides` 透传
- `pce/server.py`：`serve()` 去掉错误的 `pce_root/.env` 加载（uvx 环境下指向 cache 目录）；`_bootstrap()` 加载项目 `.env`（`override=False`，优先级低于 MCP config env）
- README：更新环境变量表（新增 `PCE_API_KEY`/`PCE_API_BASE`，`PCE_MODEL` 标注必填）；补充多供应商 MCP config 示例；MCP 配置节加 env 示例
- 单元测试 43/43 通过
下一步：CLI/playground 文档完善 或 性能调优

### 2026-03-18（本轮）
本轮完成：test_e2e.py 适配 pce_init 新接口
- 去掉 PCEAgent / InsightCache / SerenaClient 外部构造，改为 `PCEContext()` + `handle_init()`
- 去掉 serena_path 读取与校验（Serena 由 uvx 内部拉起）
- `_cleanup` 签名简化为单参数，内部检查 watcher/serena_client 是否 None
- 新增初始化失败快速退出（`if not init_result.get("initialized")`）
- 单元测试 43/43 通过
下一步：CLI/playground 文档完善 或 性能调优

### 2026-03-18（本轮）
本轮完成：README 全面更新（uvx 一键安装 + pce_init 工具说明）
- 安装节：去掉 git clone + uv sync，改为 uvx 按需运行方式
- MCP 配置：command 改为 uvx，去掉 PCE_PROJECT_PATH env
- 补充 pce_init 工具文档（会话开始时首先调用、幂等/路径冲突说明）
- 使用示例：明确 pce_init 为第一步，补充未初始化时写工具也不可用
- 环境变量表：删除 PCE_PROJECT_PATH / SERENA_PATH，PCE_SERENA_TIMEOUT 默认值改为 180
- 技术架构图：pce_init 加入 MCP 入口，Bootstrap 改为 pce_init 驱动描述
下一步：test_e2e.py 适配 pce_init 新接口

### 2026-03-18 08:43
本轮完成：一键安装改造全量实施（pce_init 工具 + uvx Serena + 空壳启动）
主体更新："已完成"（服务层/基础层/cli）、"待完成"（移除已完成项，补 README/e2e 适配）、"架构速查"（pce_init 流程 + 状态机）、"关键决策"（新增 8 条）、"项目概况"（补 pce_init 工具）
下一步：更新 README 安装说明；test_e2e.py 适配 pce_init 新接口

### 2026-03-18 06:52
本轮完成：README 全面更新 + 一键安装方案调研
主体更新："待完成"（Serena uvx 改造升级为一键安装方案，含具体改动点清单）
下一步：实施一键安装改造（`serena_client.py` uvx 启动 + 去掉 `serena_install_path` 链路）

### 2026-03-18 06:00
本轮完成：Serena 初始化流程完善（eager bootstrap + activate_project 校验 + Event 门控）
主体更新："已完成"（服务层 _bootstrap/bootstrap_event/warnings）、"关键决策"（bootstrap 策略/激活校验/onboarding 分离）、"待完成"（移除 Serena 初始化待办）
下一步：CLI/playground 文档完善 或 性能调优

### 2026-03-18 02:50
本轮完成：e2e 集成测试跑通 + 模型降级路由机制
主体更新："已完成"（Agent 层 _completion/LLMCompletionError/marker）、"测试工具"（test_e2e.py、43/43）、"关键决策"（降级策略/异常分类/Timeout 处理）、"待完成"（移除集成测试待办）
下一步：CLI/playground 文档完善 或 性能调优（并发 spawn、InsightCache）

### 2026-03-16
架构审查：确认 PCE 与 Serena 零 import 耦合（仅 MCP stdio 协议交互），唯一绑定点为 `serena_client.py:259` 的启动命令。
记录待办：公开发布前将 Serena 启动方式从本地 clone + `uv run --project` 改为 `uvx --from git+https://github.com/oraios/serena`，去掉 `serena_install_path` 参数。

### 2026-03-13 06:40
本轮完成：spawn 路径测试覆盖（T16~T20）+ 服务层全量接线
主体更新："已完成"（测试工具/服务层/mock_tool_provider）、"待完成"（清空）、"架构速查"（pce_sync 流程）、"关键决策"（sweep_stale 位置）
下一步：集成测试（真实 Serena + LLM）或性能调优

### 2026-03-13
本轮完成：SubAgent spawn 架构全量实现（commit 8383632）
- 新增 `pce/agent_runtime/`（contracts.py + spawner.py）
- `pce/agent.py` 集成 spawn 分拣、deadline、递归保护、InsightCache 写入
- SYSTEM_PROMPT_HEADER 新增 spawn_agent 使用策略
- _CHILD_SYSTEM_PROMPT 扩展为 5 条规则
- 全量测试 24/24 通过
下一步：补 spawn 路径测试用例 → 服务层接线

### 2026-03-06 08:14
本轮完成：InsightCache 集成到 PCEAgent（deliver 后蒸馏写入 + system prompt 注入）
- 测试 24/24 通过

### 2026-03-06 07:31
本轮完成：实现 InsightCache 持久化认知缓存（Step 1）
