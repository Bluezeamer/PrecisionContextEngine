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
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
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

    markdown: NonEmptyStr = Field(..., description="Markdown 格式的查询结果")

    @property
    def answer(self) -> str:
        """兼容旧调用点，返回 markdown 正文。"""
        return self.markdown


class ImpactRequest(BaseSchema):
    """pce_impact 工具的请求参数。"""

    target: NonEmptyStr = Field(..., description="影响分析目标（符号名或文件路径）")
    depth: int = Field(default=3, ge=1, le=10, description="引用链追踪深度")
    max_nodes: int = Field(default=200, ge=1, le=2000, description="最大分析节点数")


class ImpactResponse(BaseSchema):
    """pce_impact 工具的响应结果。"""

    markdown: NonEmptyStr = Field(..., description="Markdown 格式的影响分析结果")

    @property
    def answer(self) -> str:
        """兼容旧调用点，返回 markdown 正文。"""
        return self.markdown


# ============================================================================
# DigestDelta / 模块 Registry
# ============================================================================


class SymbolFact(BaseSchema):
    """轻量符号事实,用于差异事实包。"""

    name: NonEmptyStr = Field(..., description="符号名")
    kind: SymbolKind = Field(..., description="符号类型")
    line_start: int = Field(..., ge=1, description="起始行号")
    line_end: int = Field(..., ge=1, description="结束行号")

    @model_validator(mode="after")
    def _validate_line_range(self) -> SymbolFact:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class PatchBlock(BaseSchema):
    """代码块级 before/after 对照。"""

    old_start: int | None = Field(default=None, ge=1, description="旧代码起始行")
    old_end: int | None = Field(default=None, ge=1, description="旧代码结束行")
    new_start: int | None = Field(default=None, ge=1, description="新代码起始行")
    new_end: int | None = Field(default=None, ge=1, description="新代码结束行")
    old_snippet: str = Field(default="", description="旧代码片段")
    new_snippet: str = Field(default="", description="新代码片段")


class ChangedFileFact(BaseSchema):
    """模块差异事实包中的单文件事实。"""

    path: Path = Field(..., description="相对项目根的文件路径")
    status: Literal["modified", "created", "deleted"] = Field(..., description="文件变更状态")
    old_hash: str | None = Field(default=None, description="旧文件哈希")
    new_hash: str | None = Field(default=None, description="新文件哈希")
    old_content: str | None = Field(default=None, description="旧文件内容")
    new_content: str | None = Field(default=None, description="新文件内容")
    old_symbols: list[SymbolFact] = Field(default_factory=list, description="旧文件符号概览")
    new_symbols: list[SymbolFact] = Field(default_factory=list, description="新文件符号概览")
    patch_blocks: list[PatchBlock] = Field(default_factory=list, description="代码块差异列表")


class FileBaseline(BaseSchema):
    """文件 baseline 快照。"""

    path: Path = Field(..., description="相对项目根的文件路径")
    content_hash: NonEmptyStr = Field(..., description="文件内容哈希")
    content: str = Field(default="", description="文件基线内容")
    symbols: list[SymbolFact] = Field(default_factory=list, description="文件基线符号概览")
    captured_at: UTCDateTime = Field(..., description="基线采集时间（UTC）")


class ModuleDigestDelta(BaseSchema):
    """模块级认知修正事实包。"""

    module_id: UUIDStr = Field(..., description="模块稳定 ID")
    module_slug: NonEmptyStr = Field(..., description="模块稳定 slug")
    module_name: NonEmptyStr = Field(..., description="模块展示名")
    annotation_baseline: str = Field(default="", description="当前模块认知基线")
    related_insights: list[InsightFact] = Field(
        default_factory=list, description="与该模块相关的 insight"
    )
    changed_files: list[ChangedFileFact] = Field(
        default_factory=list, description="该模块相关的文件差异事实"
    )
    external_context: list[str] = Field(
        default_factory=list, description="模块级外部上下文提示"
    )


class ModuleRecord(BaseSchema):
    """模块 Registry 中的单条记录。"""

    module_id: UUIDStr = Field(..., description="模块稳定 ID")
    slug: NonEmptyStr = Field(..., description="模块稳定 slug")
    display_name: NonEmptyStr = Field(..., description="模块展示名")
    file_paths: list[NonEmptyStr] = Field(default_factory=list, description="模块覆盖文件")
    historical_file_paths: list[NonEmptyStr] = Field(
        default_factory=list,
        description="模块历史覆盖文件（包含已删除或已迁移路径）",
    )
    key_symbols: list[NonEmptyStr] = Field(default_factory=list, description="模块关键符号")
    created_at: UTCDateTime = Field(..., description="创建时间（UTC）")
    updated_at: UTCDateTime = Field(..., description="最近更新时间（UTC）")
    aliases: list[NonEmptyStr] = Field(default_factory=list, description="历史展示名")
    status: Literal["active", "merged", "deprecated"] = Field(
        default="active", description="模块状态"
    )
    area_slug: Optional[str] = Field(
        default=None,
        description="当前树状认知导航归属区域 slug（缓存字段，非稳定 identity）",
    )


class ModuleRegistry(BaseSchema):
    """模块稳定 identity 注册表。"""

    version: str = Field(default="1", description="Registry 版本号")
    records: dict[str, ModuleRecord] = Field(
        default_factory=dict, description="模块记录映射（key 为 module_id）"
    )


# ============================================================================
# 树状认知导航
# ============================================================================


class AreaRecord(BaseSchema):
    """树状认知导航中的区域记录。"""

    slug: NonEmptyStr = Field(..., description="区域稳定 slug")
    display_name: NonEmptyStr = Field(..., description="区域展示名")
    summary: str = Field(default="", description="区域一句话摘要")
    module_slugs: list[NonEmptyStr] = Field(
        default_factory=list, description="挂载到该区域的模块 slug 列表"
    )
    recommended_order: list[NonEmptyStr] = Field(
        default_factory=list, description="区域内推荐阅读顺序（module slug 列表）"
    )
    source_prefixes: list[str] = Field(
        default_factory=list, description="该区域对应的目录/路径前缀线索"
    )
    is_fallback: bool = Field(default=False, description="是否为兜底区域")

    @model_validator(mode="before")
    @classmethod
    def _compat_title(cls, values: Any) -> Any:
        """兼容旧代码中使用 title= 创建 AreaRecord 的场景。"""
        if isinstance(values, dict) and "title" in values and "display_name" not in values:
            values["display_name"] = values.pop("title")
        return values

    @property
    def title(self) -> str:
        """兼容旧代码中的 area.title 访问。"""
        return self.display_name


class NavigationTree(BaseSchema):
    """三层认知导航的结构树（项目→区域→模块）。"""

    version: str = Field(default="2", description="结构树版本号")
    generated_at: UTCDateTime = Field(..., description="最近生成时间（UTC）")
    project_summary: str = Field(default="", description="项目级一句话概览")
    fallback_area_slug: NonEmptyStr = Field(
        default="fallback", description="fallback 区域的 slug"
    )
    source_digest: str = Field(
        default="", description="源模块集合指纹（Python 侧 SHA-256 计算）"
    )
    areas: list[AreaRecord] = Field(default_factory=list, description="区域列表")

    @model_validator(mode="after")
    def _validate_fallback_area(self) -> "NavigationTree":
        """校验 fallback 区域必须唯一且与 fallback_area_slug 一致。"""
        fallback_areas = [a for a in self.areas if a.is_fallback]
        if len(fallback_areas) != 1:
            raise ValueError("navigation tree must contain exactly one fallback area")
        if fallback_areas[0].slug != self.fallback_area_slug:
            raise ValueError(
                f"fallback_area_slug ({self.fallback_area_slug}) "
                f"must match the fallback area record ({fallback_areas[0].slug})"
            )
        return self


# 兼容旧引用
AnnotationTree = NavigationTree


# ============================================================================
# Insight Cache
# ============================================================================


class InsightConfidence(str, Enum):
    """Insight 条目置信度枚举。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InsightFact(BaseSchema):
    """供 DigestDelta 使用的轻量 insight 事实。"""

    id: UUIDStr = Field(..., description="Insight 条目 ID")
    scope: NonEmptyStr = Field(..., description="Insight 作用域")
    content: NonEmptyStr = Field(..., description="Insight 正文")
    confidence: InsightConfidence = Field(..., description="Insight 置信度")
    created_at: UTCDateTime = Field(..., description="Insight 创建时间（UTC）")


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
