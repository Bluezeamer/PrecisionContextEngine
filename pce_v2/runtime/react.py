from __future__ import annotations

import json
import time
from typing import Any

import litellm

from pce._env import (
    build_litellm_model,
    configure_litellm_runtime,
    get_agent_timeout,
    get_completion_overrides,
    get_completion_retries_per_model,
    get_env_text,
    get_temperature,
)
from pce.serena_client import SerenaClientError

from ..contracts import ImpactRequest, PreparedExecution, QueryRequest
from ..engine.session import ProjectSession
from ..integrations.serena import SerenaAdapter
from ..runtime.executor import QueryImpactExecutor
from ..tools.gateway import ReadOnlyToolGateway

DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_NO_TOOL_RETRIES = 2


class MinimalReActRuntime:
    """v2 最小 ReAct runtime。

    特化目标：只保证 query / impact 主线可执行，不做通用 agent 平台。
    """

    def __init__(self) -> None:
        configure_litellm_runtime()
        self._executor = QueryImpactExecutor()
        self._gateway = ReadOnlyToolGateway()
        self._model = build_litellm_model(
            get_env_text("PCE_PROVIDER"),
            get_env_text("PCE_MODEL") or "gpt-4o-mini",
        )
        self._fallback_model = self._build_fallback_model()
        self._completion_overrides = get_completion_overrides()
        self._temperature = get_temperature(specific_key=None, default=0.2)
        self._timeout_seconds = get_agent_timeout(DEFAULT_TIMEOUT_SECONDS)
        self._retries_per_model = get_completion_retries_per_model(3)

    async def run_query(self, session: ProjectSession, request: QueryRequest) -> dict[str, Any]:
        prepared = self._executor.prepare_query(session, request)
        return await self._run(session, prepared, user_prompt=request.question)

    async def run_impact(self, session: ProjectSession, request: ImpactRequest) -> dict[str, Any]:
        user_prompt = f"target={request.target}\nchange_type={request.change_type}"
        if request.file:
            user_prompt += f"\nfile={request.file}"
        prepared = self._executor.prepare_impact(session, request)
        return await self._run(session, prepared, user_prompt=user_prompt)

    async def _run(
        self,
        session: ProjectSession,
        prepared: PreparedExecution,
        *,
        user_prompt: str,
    ) -> dict[str, Any]:
        start = time.monotonic()
        deadline = start + self._timeout_seconds
        request_id = session.trace_store.new_request_id()
        tools = self._gateway.build_openai_tools(prepared)
        session.trace_store.emit(
            request_id=request_id,
            mode=prepared.request.mode.value,
            event="request_start",
            payload={
                "tool_count": str(len(tools)),
                "max_tool_calls": str(prepared.request.budget.max_tool_calls),
            },
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prepared.prompt.system},
            {
                "role": "user",
                "content": user_prompt + "\n\n" + "\n\n".join(prepared.prompt.context_blocks),
            },
        ]
        tool_calls_used = 0
        no_tool_retries = 0

        try:
            async with SerenaAdapter(session.project_root) as serena:
                while time.monotonic() < deadline:
                    session.trace_store.emit(
                        request_id=request_id,
                        mode=prepared.request.mode.value,
                        event="completion_start",
                        payload={"message_count": str(len(messages))},
                    )
                    response = await self._complete(messages, tools)
                    message = response["choices"][0]["message"]
                    session.trace_store.emit(
                        request_id=request_id,
                        mode=prepared.request.mode.value,
                        event="completion_done",
                        payload={
                            "tool_calls": str(len(message.get("tool_calls") or [])),
                            "has_content": str(bool(message.get("content"))).lower(),
                        },
                    )
                    assistant_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": message.get("content") or "",
                    }
                    if message.get("tool_calls"):
                        assistant_message["tool_calls"] = message["tool_calls"]
                    messages.append(assistant_message)

                    tool_calls = message.get("tool_calls") or []
                    if not tool_calls:
                        no_tool_retries += 1
                        if no_tool_retries > MAX_NO_TOOL_RETRIES:
                            session.trace_store.emit(
                                request_id=request_id,
                                mode=prepared.request.mode.value,
                                event="request_end",
                                payload={"status": "failed", "reason": "no_tool_exhausted"},
                            )
                            return {
                                "ok": False,
                                "error": "模型未能按工具协议完成任务",
                                "mode": prepared.request.mode.value,
                                "request_id": request_id,
                            }
                        messages.append(
                            {
                                "role": "user",
                                "content": "必须使用工具继续取证，并最终调用 deliver 结束任务。",
                            }
                        )
                        continue

                    no_tool_retries = 0
                    for tool_call in tool_calls:
                        tool_calls_used += 1
                        tool_name = tool_call["function"]["name"]
                        arguments = self._parse_arguments(tool_call["function"].get("arguments"))
                        if tool_name == "deliver":
                            session.trace_store.emit(
                                request_id=request_id,
                                mode=prepared.request.mode.value,
                                event="deliver",
                                payload={
                                    "tool_calls_used": str(tool_calls_used),
                                    "confidence": str(arguments.get("confidence") or "medium"),
                                },
                            )
                            session.trace_store.emit(
                                request_id=request_id,
                                mode=prepared.request.mode.value,
                                event="request_end",
                                payload={"status": "ok"},
                            )
                            return {
                                "ok": True,
                                "mode": prepared.request.mode.value,
                                "answer": str(arguments.get("answer", "")).strip(),
                                "confidence": str(arguments.get("confidence") or "medium"),
                                "tool_calls_used": tool_calls_used,
                                "request_id": request_id,
                            }
                        session.trace_store.emit(
                            request_id=request_id,
                            mode=prepared.request.mode.value,
                            event="tool_call_start",
                            payload={"tool_name": tool_name},
                        )
                        try:
                            tool_result = await self._gateway.execute_tool(
                                session,
                                name=tool_name,
                                arguments=arguments,
                                max_result_chars=prepared.request.budget.max_result_chars,
                                serena=serena,
                            )
                        except SerenaClientError as exc:
                            tool_result = (
                                f"工具调用失败: {type(exc).__name__}: {exc}\n\n"
                                "请改用现有证据或其他工具继续，并尽快 deliver。"
                            )
                        session.trace_store.emit(
                            request_id=request_id,
                            mode=prepared.request.mode.value,
                            event="tool_call_done",
                            payload={
                                "tool_name": tool_name,
                                "result_chars": str(len(tool_result)),
                            },
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "name": tool_name,
                                "content": tool_result,
                            }
                        )
                        if tool_calls_used >= prepared.request.budget.max_tool_calls:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": "工具预算已接近上限。请基于现有证据调用 deliver 结束任务。",
                                }
                            )
        except SerenaClientError as exc:
            session.trace_store.emit(
                request_id=request_id,
                mode=prepared.request.mode.value,
                event="request_end",
                payload={"status": "failed", "reason": f"serena_init:{type(exc).__name__}"},
            )
            return {
                "ok": False,
                "error": f"Serena 初始化失败: {exc}",
                "mode": prepared.request.mode.value,
                "request_id": request_id,
            }
        session.trace_store.emit(
            request_id=request_id,
            mode=prepared.request.mode.value,
            event="request_end",
            payload={"status": "failed", "reason": "timeout"},
        )
        return {
            "ok": False,
            "error": "运行超时",
            "mode": prepared.request.mode.value,
            "request_id": request_id,
        }

    async def _complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        attempts: list[str] = []
        for model in self._candidate_models():
            for _ in range(self._retries_per_model):
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=self._temperature,
                        **self._completion_overrides,
                    )
                    if hasattr(response, "model_dump"):
                        return response.model_dump()
                    return response
                except Exception as exc:  # noqa: BLE001
                    attempts.append(f"{model}: {type(exc).__name__}: {exc}")
                    continue
        raise RuntimeError("LLM completion failed: " + " | ".join(attempts))

    def _candidate_models(self) -> list[str]:
        candidates = [self._model]
        if self._fallback_model and self._fallback_model != self._model:
            candidates.append(self._fallback_model)
        return candidates

    def _build_fallback_model(self) -> str | None:
        raw = get_env_text("PCE_FALLBACK_MODEL")
        if not raw:
            return None
        return build_litellm_model(get_env_text("PCE_PROVIDER"), raw)

    @staticmethod
    def _parse_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                return {}
            return value if isinstance(value, dict) else {}
        return {}
