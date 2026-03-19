"""PCE 数据模型定义。

本模块定义了 PCE 系统中所有核心数据结构,使用 Pydantic v2 提供类型安全和数据验证。

核心设计原则:
1. 所有时间字段统一使用 UTC 时区
2. 所有路径字段使用 pathlib.Path
3. ID 字段使用 UUID 字符串格式
4. 字符串字段自动去除首尾空格
5. 提供清晰的字段描述便于理解
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


def _ensure_utc(value: datetime) -> datetime:
    """确保 datetime 对象带有 UTC 时区信息。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_uuid(value: str) -> str:
    """验证字符串是否为合法的 UUID 格式。"""
    try:
        uuid.UUID(value)
    except ValueError as e:
        raise ValueError(f"Invalid UUID format: {value}") from e
    return value


# 类型别名定义
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
UUIDStr = Annotated[NonEmptyStr, AfterValidator(_validate_uuid)]
UTCDateTime = Annotated[datetime, AfterValidator(_ensure_utc)]


class BaseSchema(BaseModel):
    """所有模型的基类,禁止额外字段以保证数据严格性。"""

    model_config = ConfigDict(extra="forbid", frozen=False)


# ============================================================================
# 项目与文件元数据
# ============================================================================


class ProjectMeta(BaseSchema):
    """项目元数据,记录项目基本信息和索引统计。"""

    root_path: Path = Field(..., description="项目根路径")
    created_at: UTCDateTime = Field(..., description="项目首次索引时间（UTC）")
    index_version: NonEmptyStr = Field(..., description="索引版本号")
    file_count: int = Field(..., ge=0, description="文件总数")
    loc_total: int = Field(..., ge=0, description="代码行总数")


class FileMeta(BaseSchema):
    """文件元数据,记录单个文件的基本信息。"""

    path: Path = Field(..., description="文件路径")
    language: NonEmptyStr = Field(..., description="语言标识(如 python, typescript)")
    size_bytes: int = Field(..., ge=0, description="文件大小（字节）")
    mtime: UTCDateTime = Field(..., description="最后修改时间（UTC）")
    loc: int = Field(..., ge=0, description="代码行数（不含空行和注释）")


# ============================================================================
# 符号与引用
# ============================================================================


class SymbolKind(str, Enum):
    """符号类型枚举。"""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    MODULE = "module"
    IMPORT = "import"


class SymbolRef(BaseSchema):
    """符号引用,记录代码中的一个符号定义。"""

    symbol_id: UUIDStr = Field(..., description="符号唯一 ID（UUID）")
    name: NonEmptyStr = Field(..., description="符号名称")
    kind: SymbolKind = Field(..., description="符号类型")
    file_path: Path = Field(..., description="定义所在文件路径")
    line_start: int = Field(..., ge=1, description="起始行号（从 1 开始）")
    line_end: int = Field(..., ge=1, description="结束行号（从 1 开始）")
    signature: Optional[NonEmptyStr] = Field(
        default=None, description="函数/方法签名（可选）"
    )

    @model_validator(mode="after")
    def _validate_line_range(self) -> SymbolRef:
        """验证行号范围的合法性。"""
        if self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )
        return self


class ReferenceRelation(str, Enum):
    """引用关系类型枚举。"""

    CALLS = "calls"  # 函数调用
    IMPORTS = "imports"  # 模块导入
    USES = "uses"  # 变量/类型使用
    INHERITS = "inherits"  # 继承关系


class ReferenceEdge(BaseSchema):
    """引用边,表示两个符号之间的引用关系。"""

    from_symbol_id: UUIDStr = Field(..., description="引用起点符号 ID")
    to_symbol_id: UUIDStr = Field(..., description="引用终点符号 ID")
    relation: ReferenceRelation = Field(..., description="引用关系类型")
    evidence: NonEmptyStr = Field(..., description="证据（行号或代码片段摘要）")


# ============================================================================
# 索引结构
# ============================================================================


class IndexEntry(BaseSchema):
    """单个文件的索引条目,包含文件元数据、符号列表和引用关系。"""

    file_meta: FileMeta = Field(..., description="文件元数据")
    symbols: list[SymbolRef] = Field(default_factory=list, description="符号列表")
    imports: list[NonEmptyStr] = Field(default_factory=list, description="导入列表")
    edges: list[ReferenceEdge] = Field(default_factory=list, description="引用边列表")


class BuildStats(BaseSchema):
    """索引构建统计信息。"""

    total_files: int = Field(..., ge=0, description="处理文件总数")
    total_symbols: int = Field(..., ge=0, description="符号总数")
    total_edges: int = Field(..., ge=0, description="引用边总数")
    duration_ms: int = Field(..., ge=0, description="构建耗时（毫秒）")
    warnings: list[NonEmptyStr] = Field(default_factory=list, description="构建警告列表")


class IndexSnapshot(BaseSchema):
    """完整的项目索引快照。

    注意: created_at 表示快照生成时间,project_meta.created_at 表示项目首次索引时间。
    """

    project_meta: ProjectMeta = Field(..., description="项目元数据")
    entries: list[IndexEntry] = Field(default_factory=list, description="索引条目列表")
    created_at: UTCDateTime = Field(..., description="快照生成时间（UTC,每次重建时更新）")
    build_stats: BuildStats = Field(..., description="构建统计信息")


# ============================================================================
# Memory 与会话
# ============================================================================


class MemoryItemType(str, Enum):
    """记忆条目类型枚举。"""

    SUMMARY = "summary"  # 摘要
    DECISION = "decision"  # 决策记录
    QUERY = "query"  # 查询记录
    IMPACT = "impact"  # 影响分析记录
    NOTE = "note"  # 备注


class MemoryItem(BaseSchema):
    """记忆条目,存储 Agent 的推理过程和结论。"""

    item_id: UUIDStr = Field(..., description="记忆条目唯一 ID（UUID）")
    item_type: MemoryItemType = Field(..., description="记忆类型")
    content: NonEmptyStr = Field(..., description="记忆内容")
    tags: list[NonEmptyStr] = Field(default_factory=list, description="标签列表")
    related_files: list[Path] = Field(default_factory=list, description="关联文件路径列表")
    created_at: UTCDateTime = Field(..., description="创建时间（UTC）")

    @field_validator("tags")
    @classmethod
    def _dedupe_tags(cls, value: list[str]) -> list[str]:
        """去除标签列表中的重复项,保持顺序。"""
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


class SessionState(BaseSchema):
    """会话状态,记录单个对话会话的上下文。"""

    session_id: UUIDStr = Field(..., description="会话唯一 ID（UUID）")
    started_at: UTCDateTime = Field(..., description="会话开始时间（UTC）")
    last_step: Optional[NonEmptyStr] = Field(
        default=None, description="最后执行的步骤描述"
    )
    last_error: Optional[NonEmptyStr] = Field(
        default=None, description="最后发生的错误信息"
    )


# ============================================================================
# MCP 工具协议
# ============================================================================


class ErrorResponse(BaseSchema):
    """统一的错误响应格式。"""

    success: bool = Field(default=False, description="操作是否成功（错误时固定为 False）")
    error_code: NonEmptyStr = Field(..., description="错误代码（如 INDEX_NOT_FOUND）")
    error_message: NonEmptyStr = Field(..., description="人类可读的错误信息")
    details: Optional[NonEmptyStr] = Field(default=None, description="详细错误信息（可选）")


class InitResponse(BaseModel):
    """pce_init 工具的响应结果。"""

    initialized: bool
    status: Literal["initialized", "already_initialized", "init_failed"]
    project_path: str
    project_name: str
    file_count: int
    init_mode: Literal["full_build", "reused", "retry_after_failure"]
    warnings: list[str]
    error: str | None = None


class QueryRequest(BaseSchema):
    """pce_query 工具的请求参数。"""

    query: NonEmptyStr = Field(..., description="自然语言查询")
    top_k: int = Field(default=5, ge=1, le=50, description="返回结果数量上限")


class QueryResponse(BaseSchema):
    """pce_query 工具的响应结果。"""

    answer: NonEmptyStr = Field(..., description="查询答案正文")
    evidence: list[NonEmptyStr] = Field(default_factory=list, description="证据摘要列表")
    related_symbols: list[SymbolRef] = Field(default_factory=list, description="相关符号列表")
    related_files: list[Path] = Field(default_factory=list, description="相关文件路径列表")


class ImpactRequest(BaseSchema):
    """pce_impact 工具的请求参数。"""

    target: NonEmptyStr = Field(..., description="影响分析目标（符号名或文件路径）")
    depth: int = Field(default=3, ge=1, le=10, description="引用链追踪深度")
    max_nodes: int = Field(default=200, ge=1, le=2000, description="最大分析节点数")


class ImpactResponse(BaseSchema):
    """pce_impact 工具的响应结果。"""

    impact_chain: list[ReferenceEdge] = Field(default_factory=list, description="完整影响链路")
    boundary: list[SymbolRef] = Field(default_factory=list, description="影响边界符号列表")
    risks: list[NonEmptyStr] = Field(default_factory=list, description="风险提示列表")
    unknowns: list[NonEmptyStr] = Field(default_factory=list, description="不确定项列表")


class StatusResponse(BaseSchema):
    """pce_status 工具的响应结果。"""

    last_index_time: Optional[UTCDateTime] = Field(
        default=None, description="最后索引时间（UTC）"
    )
    index_version: Optional[NonEmptyStr] = Field(default=None, description="索引版本号")
    memory_items_count: int = Field(..., ge=0, description="记忆条目数量")
    status_message: NonEmptyStr = Field(..., description="状态描述信息")
    insight_stats: Optional[InsightStats] = Field(
        default=None, description="Insight Cache 统计信息（可选）"
    )


# ============================================================================
# Insight Cache
# ============================================================================


class InsightConfidence(str, Enum):
    """Insight 条目置信度枚举。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightEntry(BaseSchema):
    """Insight 完整条目，用于写入 entries/{uuid}.json。"""

    id: UUIDStr = Field(..., description="条目唯一 ID（UUID）")
    scope: NonEmptyStr = Field(..., description="作用域（相对项目根的文件路径）")
    content: NonEmptyStr = Field(..., description="蒸馏后的模块级理解内容")
    confidence: InsightConfidence = Field(
        default=InsightConfidence.MEDIUM, description="条目置信度"
    )
    created_at: UTCDateTime = Field(..., description="创建时间（UTC）")
    last_referenced_at: UTCDateTime = Field(..., description="最后引用时间（UTC）")
    source_hash: NonEmptyStr = Field(..., description="创建时源文件的 SHA256 哈希")
    supersedes: Optional[UUIDStr] = Field(
        default=None, description="被当前条目覆盖的旧条目 ID（可选）"
    )


class InsightIndexRecord(BaseSchema):
    """Insight 索引记录，不含 content，用于 index.json。"""

    id: UUIDStr = Field(..., description="条目唯一 ID（UUID）")
    scope: NonEmptyStr = Field(..., description="作用域（相对项目根的文件路径）")
    confidence: InsightConfidence = Field(..., description="条目置信度")
    created_at: UTCDateTime = Field(..., description="创建时间（UTC）")
    last_referenced_at: UTCDateTime = Field(..., description="最后引用时间（UTC）")
    source_hash: NonEmptyStr = Field(..., description="创建时源文件的 SHA256 哈希")
    supersedes: Optional[UUIDStr] = Field(
        default=None, description="被当前条目覆盖的旧条目 ID（可选）"
    )
    stale: bool = Field(default=False, description="是否已过时")
    stale_checked_at: Optional[UTCDateTime] = Field(
        default=None, description="最近一次 stale 检查时间（UTC）"
    )


class InsightIndex(BaseSchema):
    """Insight Cache 索引文件结构。"""

    version: str = Field(default="1", description="索引版本号")
    updated_at: UTCDateTime = Field(..., description="索引更新时间（UTC）")
    records: dict[str, InsightIndexRecord] = Field(
        default_factory=dict, description="索引记录映射（key 为条目 ID）"
    )


class InsightStats(BaseSchema):
    """Insight Cache 统计信息（用于 pce_status）。"""

    total_entries: int = Field(..., ge=0, description="条目总数")
    active_entries: int = Field(..., ge=0, description="活跃条目数")
    stale_entries: int = Field(..., ge=0, description="过时条目数")
    last_updated: Optional[UTCDateTime] = Field(default=None, description="最后更新时间（UTC）")
