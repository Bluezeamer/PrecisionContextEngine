"""工具提供者协议定义。

定义 Agent 运行时对工具层的最小抽象接口。
SerenaClient 已天然满足此协议（结构性子类型），无需显式继承。
MockToolProvider 显式实现此协议用于离线测试。

此协议将在 SubAgent 架构中作为 agent runtime 的标准工具接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    """Agent 运行时的工具提供者协议。

    任何满足以下接口的对象均可作为 PCEAgent 的工具源：
    - tools_schema: 返回工具定义列表（OpenAI function calling 格式）
    - call: 异步调用指定工具
    - connected: 连接状态
    - project_path: 当前项目路径

    异常约定：call() 在工具调用失败时应抛出 SerenaClientError 或其子类，
    以兼容 agent.py 中 _invoke_tool 的 except SerenaClientError 捕获。
    """

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        """返回工具 schema 列表（OpenAI function calling 格式）。"""
        ...

    @property
    def connected(self) -> bool:
        """是否处于可用状态。"""
        ...

    @property
    def project_path(self) -> Path:
        """当前项目根路径。"""
        ...

    async def call(self, tool_name: str, args: dict[str, Any]) -> Any:
        """调用指定工具并返回结果。

        Args:
            tool_name: 工具名称
            args: 工具参数字典

        Returns:
            工具调用结果（JSON 可序列化）

        Raises:
            SerenaClientError: 工具调用失败
        """
        ...
