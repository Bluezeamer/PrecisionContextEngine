# Sync / Insight 重构计划

最后更新时间：2026-03-30

## Compact 提示

如果对话被 compact 或切窗后上下文变薄，先回读本文件：

- `pce/docs/sync_insight_refactor_plan.md`

不要直接继续实现，先按本文恢复计划与边界，避免实现错位。

## 目标

本轮只做两件事：

1. **`pce_sync` 对齐到统一后处理链**
2. **`insight` 改为原始问答存储，不做 Python 解释层加工**

## 已澄清原则

### 1. `sync` 与 `init` 的后处理本质相同

- `init` 和 `sync` 后半段都在做：
  - dirty file / insight 的后处理判断
  - digest 是否需要执行
  - 认知是否需要内化或清理
- 因此不应在 `sync` 再维护一套独立判断逻辑
- 最合理方式是：
  - 抽一个统一的后处理入口
  - `init` 与 `sync` 都调用它

### 2. `sync` 跳过条件要收紧

当前问题：
- `dirty.empty` 时，`handle_sync()` 会直接快速返回
- 导致“**无 dirty 但有 active insights**”不会进入 digest

目标改为：
- **无 dirty 且无 insight** → 才跳过
- 否则 → 进入统一后处理链

### 3. `insight` 不再做 Python 解释层加工

不要再由 Python：
- 提取 scope
- 截断蒸馏 content
- 基于 scope 覆盖写入

改为原始问答存储：

```json
{
  "id": "...",
  "created_at": "...",
  "question": "...",
  "answer": "...",
  "confidence": "high|medium|low"
}
```

### 4. `insight` 的相关性判断后移到 Agent

- `digest` 收到的 insight 应尽量简单、原始
- 是否与 dirty files 有关、应落在哪层认知，由后续 agent 自己判断
- Python 侧只负责：
  - 存储
  - 读取
  - 预算/围栏

## 计划中的修改

### A. `sync` / `init` 后处理对齐

新增统一后处理函数（命名待实现时确定），职责：

- 输入：
  - `project_root`
  - `serena_client`
  - `insight_cache`
  - `dirty_state`
- 输出：
  - digest 是否执行
  - warnings
  - summary / stats（如需要）

两边接法：

- `init/bootstrap`
  - 在 `seed_initial_file_baselines_if_missing()` 之后调用统一后处理函数
- `sync`
  - 即使 `dirty.empty`，也不直接 return
  - 先判断是否仍有 active insights
  - 若有，则进入统一后处理函数
  - 仅在“无 dirty 且无 insights”时快速返回

### B. `Insight` 模型重构

重构 `pce/models.py`：

- `InsightFact`
  - 改为：`id / created_at / question / answer / confidence`
- `InsightEntry`
  - 改为：`id / created_at / question / answer / confidence`
- `InsightIndexRecord`
  - 仅保留索引所需最小元数据，不含 scope / source_hash / stale

### C. `InsightCache` 重构

重构 `pce/insight_cache.py`：

- `upsert()` 改为追加式写入（不再按 scope 覆盖）
- `get_top_k()` 改为按 `created_at` 逆序取最新问答
- `sweep_stale()` / `cleanup_stale()` 收缩为最小语义
  - `sweep_stale()` 保留为 no-op
  - `cleanup_stale()` 只做孤儿 entry 清理
- `stats()` / `get_all_records()` / `delete_by_ids()` 与新结构对齐

### D. `PCEAgent` insight 持久化重构

重构 `pce/agent.py`：

- 删除：
  - `_extract_path_candidates`
  - `_pick_insight_scopes`
  - `_distill_insight_content`
- `_persist_insights()` 改为直接保存：
  - `question`
  - `answer`
  - `confidence`

### E. `digest` insight 注入重构

重构 `pce/digest_agent.py` / `pce/digest_cognition_agent.py`：

- `_load_active_insights()` 读取新结构
- `_render_insight()` 改为渲染：
  - `id`
  - `created_at`
  - `confidence`
  - `question`
  - `answer`
- stage1 / stage2 / stageB 的 facts 包同步对齐

## 风险与取舍

### 1. 放弃 scope / stale / source_hash

这意味着：
- insight 不再由 Python 侧做文件级 stale 判断
- stale 风险完全交给后续 digest stage2 / stageC

这是符合当前设计方向的取舍：
- Python 不做解释层加工
- Agent 自己判断相关性与陈旧性

### 2. 旧 insight 不兼容

当前 `.pce/insights/` 若存在旧格式数据：
- scope/content/source_hash/stale

直接忽略，不做迁移，不做兼容读取。

## 实施顺序

1. 落计划文档（当前已完成）
2. 先对齐 `sync` / `init` 的统一后处理入口
3. 再重构 `insight` 数据模型与缓存
4. 再改 `digest` 注入格式
5. 最后补最小回归：
   - 无 dirty + 有 insight 时 `sync` 仍触发 digest
   - insight 以 `question/answer` 形式进入 digest
   - 旧 Python scope 化逻辑已移除
