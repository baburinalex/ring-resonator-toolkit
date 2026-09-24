"""Конфиг стенда: модели (OpenAI-совместимый API) и лимиты прогона."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Одна модель за OpenAI-совместимым API (Ollama, vLLM, OpenAI, ...)."""

    name: str
    model: str
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.0
    seed: int | None = 0
    api_key_env: str | None = None  # имя переменной окружения с ключом; None — без ключа
    max_tokens: int | None = None
    request_timeout_s: float = 300.0

    def public(self) -> dict:
        """Поля для лога (без секретов)."""
        return {
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 20
    time_limit_s: float = 600.0
    tool_timeout_s: float = 60.0
    max_tool_output_chars: int = 20_000
    modes: tuple[str, ...] = ("naive", "operator")


@dataclass(frozen=True)
class EvalConfig:
    models: tuple[ModelConfig, ...]
    run: RunConfig = field(default_factory=RunConfig)


def load_config(path: str | Path) -> EvalConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    models = tuple(ModelConfig(**m) for m in data.get("models", []))
    if not models:
        raise ValueError(f"{path}: нет ни одной секции [[models]]")
    run = data.get("run", {})
    if "modes" in run:
        run = {**run, "modes": tuple(run["modes"])}
    return EvalConfig(models=models, run=RunConfig(**run))
