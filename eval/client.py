"""Клиенты чат-модели: OpenAI-совместимый HTTP и заглушка для тестов.

Клиент получает список сообщений и схемы инструментов в формате OpenAI
function calling и возвращает ``ChatResponse`` — сообщение ассистента
(dict в том же формате) и расход токенов.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .config import ModelConfig


@dataclass
class ChatResponse:
    message: dict  # {"role": "assistant", "content": ..., "tool_calls": [...]}
    usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens


class ChatClient(Protocol):
    def chat(
        self, messages: list[dict], tools: list[dict], timeout_s: float | None = None
    ) -> ChatResponse: ...


class ClientError(RuntimeError):
    pass


class OpenAICompatClient:
    """POST {base_url}/chat/completions. Работает с Ollama без изменений кода."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _headers(self) -> dict:
        key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        # Ollama ключ не проверяет, но некоторые прокси требуют заголовок
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key or 'none'}"}

    def chat(
        self, messages: list[dict], tools: list[dict], timeout_s: float | None = None
    ) -> ChatResponse:
        body: dict = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if tools:
            body["tools"] = tools
        if self.cfg.seed is not None:
            body["seed"] = self.cfg.seed
        if self.cfg.max_tokens is not None:
            body["max_tokens"] = self.cfg.max_tokens

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=self._headers(), method="POST"
        )
        timeout = min(timeout_s or self.cfg.request_timeout_s, self.cfg.request_timeout_s)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise ClientError(f"HTTP {e.code} от {url}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ClientError(f"нет ответа от {url}: {e}") from e

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise ClientError(f"неожиданный ответ {url}: {str(data)[:500]}") from e
        return ChatResponse(message=message, usage=data.get("usage") or {})


class ScriptedClient:
    """Модель-заглушка: выдаёт заранее заданные ответы по порядку.

    Элемент сценария — готовое сообщение ассистента (dict) или функция
    ``messages -> dict`` (для ответов, зависящих от вывода инструментов).
    Для удобства есть ``tool_call`` и ``final``.
    """

    def __init__(self, script: Sequence[dict | Callable[[list[dict]], dict]]):
        self.script = list(script)
        self.calls: list[list[dict]] = []  # что видела модель на каждом шаге

    def chat(
        self, messages: list[dict], tools: list[dict], timeout_s: float | None = None
    ) -> ChatResponse:
        self.calls.append(json.loads(json.dumps(messages)))
        if not self.script:
            raise ClientError("сценарий заглушки исчерпан")
        step = self.script.pop(0)
        message = step(messages) if callable(step) else step
        return ChatResponse(message=message, usage={"prompt_tokens": 10, "completion_tokens": 5})

    @staticmethod
    def tool_call(name: str, arguments: dict | str, call_id: str | None = None) -> dict:
        args = arguments if isinstance(arguments, str) else json.dumps(arguments)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id or f"call_{name}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        }

    @staticmethod
    def final(content: str | dict) -> dict:
        text = content if isinstance(content, str) else json.dumps(content)
        return {"role": "assistant", "content": text}
