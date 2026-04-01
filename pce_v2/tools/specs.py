from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..contracts import StrictModel


class ToolSpec(StrictModel):
    name: str
    purpose: str
    result_policy: Literal["skeleton", "excerpt", "summary", "full"] = "excerpt"
    read_only: bool = True


class ToolAssemblyPolicy(StrictModel):
    mode: str
    allowed_tools: list[str] = Field(default_factory=list)
    denied_path_prefixes: list[str] = Field(default_factory=list)
    stable_sort: bool = True
