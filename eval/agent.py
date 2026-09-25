"""Агентный цикл с вызовом инструментов, лимитом шагов и времени."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .client import ChatClient, ClientError
from .tools import ToolContext, ToolError, ToolRegistry


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict | str
    output: str
    error: bool
    elapsed_s: float


@dataclass
class AgentResult:
    final_text: str | None
    stop_reason: str  # "final" | "max_steps" | "time_limit" | "client_error"
    steps: int
    elapsed_s: float
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [обрезано: {len(text) - limit} символов]"


def run_agent(
    client: ChatClient,
    tools: ToolRegistry,
    ctx: ToolContext,
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 20,
    time_limit_s: float = 600.0,
    max_tool_output_chars: int = 20_000,
) -> AgentResult:
    """Крутит цикл «модель -> инструменты» до финального ответа или лимита.

    Шаг — один запрос к модели. Ответ без tool_calls считается финальным.
    """
    t0 = time.monotonic()
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = AgentResult(final_text=None, stop_reason="max_steps", steps=0, elapsed_s=0.0)
    schemas = tools.schemas()

    for step in range(1, max_steps + 1):
        remaining = time_limit_s - (time.monotonic() - t0)
        if remaining <= 0:
            result.stop_reason = "time_limit"
            break
        try:
            resp = client.chat(messages, schemas, timeout_s=remaining)
        except ClientError as e:
            result.stop_reason, result.error = "client_error", str(e)
            break
        result.steps = step
        for k in ("prompt_tokens", "completion_tokens"):
            result.usage[k] += int(resp.usage.get(k) or 0)

        msg = {k: v for k, v in resp.message.items() if v is not None}
        msg["role"] = "assistant"
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            result.final_text = msg.get("content") or ""
            result.stop_reason = "final"
            break

        for call in calls:
            fn = call.get("function", {})
            name, raw_args = fn.get("name", ""), fn.get("arguments") or "{}"
            ts = time.monotonic()
            ctx.timeout_s = max(1.0, min(ctx.timeout_s, time_limit_s - (ts - t0)))
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                if not isinstance(args, dict):
                    raise ToolError("аргументы должны быть JSON-объектом")
                output, error = tools.call(name, args, ctx), False
            except json.JSONDecodeError as e:
                args, output, error = raw_args, f"ОШИБКА: аргументы не JSON: {e}", True
            except ToolError as e:
                args = args if isinstance(args, dict) else raw_args
                output, error = f"ОШИБКА: {e}", True
            output = _truncate(output, max_tool_output_chars)
            result.tool_calls.append(
                ToolCallRecord(step, name, args, output, error, time.monotonic() - ts)
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": output}
            )
    else:
        result.stop_reason = "max_steps"

    result.messages = messages
    result.elapsed_s = time.monotonic() - t0
    return result
