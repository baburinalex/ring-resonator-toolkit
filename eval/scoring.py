"""Разбор итогового ответа агента и оценка против truth.json."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from ring_toolkit.benchmark import ANALYZER_ALIASES, ANOMALY_CODES, REGIMES

ANSWER_KEYS = ("regime", "q_i", "anomalies", "confidence")


@dataclass
class Answer:
    regime: str
    q_i: float | None
    anomalies: list[str]
    confidence: float


def parse_answer(text: str | None) -> tuple[Answer | None, str | None]:
    """Строгий разбор: весь ответ — один JSON-объект с ключами ANSWER_KEYS.

    Возвращает (Answer, None) или (None, причина). Обёртки вроде ```json
    не допускаются: невалидный ответ — ошибка агента, а не повод угадывать.
    """
    if text is None:
        return None, "нет финального ответа"
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        return None, f"не JSON: {e}"
    if not isinstance(data, dict):
        return None, "ответ не JSON-объект"
    if set(data) != set(ANSWER_KEYS):
        return None, f"ключи {sorted(data)} вместо {sorted(ANSWER_KEYS)}"

    regime = data["regime"]
    if regime not in REGIMES:
        return None, f"regime {regime!r} не из {list(REGIMES)}"
    q_i = data["q_i"]
    if q_i is not None:
        if isinstance(q_i, bool) or not isinstance(q_i, int | float):
            return None, "q_i не число"
        if not math.isfinite(q_i) or q_i <= 0:
            return None, "q_i должно быть конечным положительным числом"
    anomalies = data["anomalies"]
    if not isinstance(anomalies, list) or not all(isinstance(a, str) for a in anomalies):
        return None, "anomalies должен быть списком строк"
    conf = data["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, int | float) or not 0 <= conf <= 1:
        return None, "confidence должно быть числом в [0, 1]"
    return Answer(regime, None if q_i is None else float(q_i), anomalies, float(conf)), None


def normalize_code(code: str) -> str:
    c = code.strip().upper().replace("-", "_").replace(" ", "_")
    return ANALYZER_ALIASES.get(c, c)


@dataclass
class Score:
    valid: bool
    parse_error: str | None
    regime_correct: bool
    q_i_rel_error: float | None  # None — q_i не назван или ответ невалиден
    anomaly_recall: float | None  # None — у случая нет ожидаемых аномалий
    false_anomalies: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def score(answer: Answer | None, truth: dict, parse_error: str | None = None) -> Score:
    expected = {normalize_code(c) for c in truth["expected_anomalies"]}
    if answer is None:
        return Score(
            valid=False,
            parse_error=parse_error or "невалидный ответ",
            regime_correct=False,
            q_i_rel_error=None,
            anomaly_recall=0.0 if expected else None,
            false_anomalies=[],
        )
    named = {normalize_code(c) for c in answer.anomalies}
    q_err = None if answer.q_i is None else abs(answer.q_i - truth["q_i"]) / truth["q_i"]
    return Score(
        valid=True,
        parse_error=None,
        regime_correct=answer.regime == truth["regime"],
        q_i_rel_error=q_err,
        anomaly_recall=len(named & expected) / len(expected) if expected else None,
        false_anomalies=sorted(named - expected),
    )


def known_codes() -> dict[str, str]:
    return dict(ANOMALY_CODES)
