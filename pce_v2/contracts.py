from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UTCModel(StrictModel):
    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class NavigationNodeKind(str, Enum):
    ROOT = "root"
    AREA = "area"
    MODULE = "module"


class InsightHostKind(str, Enum):
    INDEX = "index"
    AREA = "area"
    MODULE = "module"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NavigationNode(StrictModel):
    id: NonEmptyStr
    kind: NavigationNodeKind
    name: NonEmptyStr
    slug: NonEmptyStr
    parent_id: str | None = None


class CoverageRule(StrictModel):
    node_id: NonEmptyStr
    include_paths: list[NonEmptyStr] = Field(default_factory=list)
    exclude_paths: list[NonEmptyStr] = Field(default_factory=list)
    priority: int = 0


class FileBinding(StrictModel):
    file_path: NonEmptyStr
    node_id: NonEmptyStr


class NavigationTree(StrictModel):
    version: NonEmptyStr = "2"
    root_path: Path
    nodes: list[NavigationNode] = Field(default_factory=list)
    rules: list[CoverageRule] = Field(default_factory=list)
    bindings: list[FileBinding] = Field(default_factory=list)


class InsightHostRef(StrictModel):
    kind: InsightHostKind
    slug: str | None = None

    @property
    def relative_path(self) -> Path:
        if self.kind is InsightHostKind.INDEX:
            return Path("index.md")
        if self.slug is None:
            raise ValueError("area/module 宿主必须包含 slug")
        if self.kind is InsightHostKind.AREA:
            return Path("areas") / f"{self.slug}.md"
        return Path("modules") / f"{self.slug}.md"


class InsightRecordMeta(StrictModel):
    id: NonEmptyStr
    files: list[NonEmptyStr]
    created_at: datetime
    confidence: ConfidenceLevel | None = None

    @model_validator(mode="after")
    def validate_meta(self) -> "InsightRecordMeta":
        self.created_at = UTCModel._ensure_utc(self.created_at)
        if not self.files:
            raise ValueError("files 不能为空")
        return self


class InsightRecord(StrictModel):
    meta: InsightRecordMeta
    content: NonEmptyStr


class QueryRequest(StrictModel):
    question: NonEmptyStr


class ImpactRequest(StrictModel):
    target: NonEmptyStr
    change_type: Literal["modify", "rename", "delete", "add_field", "change_signature"]
    file: str | None = None


class ModeName(str, Enum):
    QUERY = "query"
    IMPACT = "impact"
    RECONCILE = "reconcile"


class BudgetPolicy(StrictModel):
    max_tool_calls: int
    max_reads: int
    max_escalations: int
    max_result_chars: int


class PromptContract(StrictModel):
    identity: NonEmptyStr
    objective: NonEmptyStr
    stop_when: list[NonEmptyStr] = Field(default_factory=list)
    output_sections: list[NonEmptyStr] = Field(default_factory=list)


class PreparedRetrievalRequest(StrictModel):
    mode: ModeName
    session_stable_context: dict[str, str]
    turn_local_context: dict[str, str]
    policy_context: dict[str, str]
    contract: PromptContract
    budget: BudgetPolicy


class AssembledPrompt(StrictModel):
    system: NonEmptyStr
    context_blocks: list[NonEmptyStr] = Field(default_factory=list)


class ToolDescriptor(StrictModel):
    name: NonEmptyStr
    purpose: NonEmptyStr
    result_policy: Literal["skeleton", "excerpt", "summary", "full"]
    read_only: bool = True


class PreparedExecution(StrictModel):
    request: PreparedRetrievalRequest
    prompt: AssembledPrompt
    tools: list[ToolDescriptor] = Field(default_factory=list)


class RuntimeIdentity(StrictModel):
    engine_id: NonEmptyStr
    session_id: NonEmptyStr
    project_root: Path


class TraceEvent(StrictModel):
    event: NonEmptyStr
    request_id: NonEmptyStr
    mode: NonEmptyStr
    timestamp: datetime
    payload: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_timestamp(self) -> "TraceEvent":
        self.timestamp = UTCModel._ensure_utc(self.timestamp)
        return self


class ReconcileRequest(StrictModel):
    dirty_files: list[NonEmptyStr] = Field(default_factory=list)


class FileBaselineRecord(StrictModel):
    file_path: NonEmptyStr
    fingerprint: NonEmptyStr
    updated_at: datetime

    @model_validator(mode="after")
    def validate_updated_at(self) -> "FileBaselineRecord":
        self.updated_at = UTCModel._ensure_utc(self.updated_at)
        return self


class ReconcileDecision(str, Enum):
    NOOP = "noop"
    PATCH_TREE = "patch_tree"
    REBUILD_TREE = "rebuild_tree"


class ReconcileResult(StrictModel):
    decision: ReconcileDecision
    changed_files: list[NonEmptyStr] = Field(default_factory=list)
    removed_insights: int = 0
    rebuilt_navigation: bool = False
    affected_areas: list[NonEmptyStr] = Field(default_factory=list)


class NavigationUpdatePlan(StrictModel):
    decision: ReconcileDecision
    affected_areas: list[NonEmptyStr] = Field(default_factory=list)
    reasons: list[NonEmptyStr] = Field(default_factory=list)
