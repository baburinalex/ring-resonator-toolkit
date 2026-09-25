"""Стенд сравнения LLM-агентов на задаче анализа спектров кольцевых резонаторов.

Вход — папка случаев от ``ring_toolkit.benchmark``. Агент работает в копии
случая без ``truth.json``; ответ сверяется с эталоном только после прогона.

Запуск:
    python -m eval run --config eval/config.example.toml --cases sweeps/benchmark
    python -m eval summarize eval_runs/
"""
