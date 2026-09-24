# eval — стенд сравнения LLM-агентов

Сравнивает модели на задаче анализа спектров кольцевых резонаторов. Случаи генерирует
`ring_toolkit.benchmark`, ответы агента сверяются с `truth.json` после прогона.

## Запуск на Ollama

```bash
# 1. Модели (любые с поддержкой tools в Ollama)
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
ollama serve                      # если ещё не запущен; API на http://localhost:11434/v1

# 2. Случаи
python -m ring_toolkit.benchmark sweeps/benchmark --seed 0          # 10 случаев
python -m ring_toolkit.benchmark sweeps/benchmark_x3 --seed 0 -n 3  # по 3 на ловушку

# 3. Прогон и сводка
cp eval/config.example.toml eval/config.toml   # поправьте список моделей
python -m eval run --config eval/config.toml --cases sweeps/benchmark --out eval_runs
cat eval_runs/summary.md

# пересобрать сводку из логов (например, после удаления части прогонов)
python -m eval summarize eval_runs
```

Другой OpenAI-совместимый сервер (vLLM, LM Studio, облачный API) подключается только
конфигом: `base_url`, `model` и, если нужен ключ, `api_key_env` — имя переменной окружения.

## Режимы

| Режим | Инструменты | Что проверяет |
|---|---|---|
| `naive` | `list_files`, `read_file`, `run_python` | может ли модель сама разобрать спектр кодом |
| `operator` | `list_files`, `run_analysis` | может ли модель правильно прочитать вывод `ring_toolkit.analyze` |

Новый инструмент: зарегистрируйте функцию в `eval/tools.py` через `@REGISTRY.tool(...)`
и добавьте её имя в `MODES`. Цикл агента менять не нужно.

## Ответ агента

Последнее сообщение — только JSON, без текста и без ```` ``` ````:

```json
{"regime": "overcoupled", "q_i": 125000, "anomalies": ["RESONANCE_SPLITTING"], "confidence": 0.7}
```

`regime` — `overcoupled` / `undercoupled` / `critical`; `q_i` — число или `null`;
`anomalies` — коды из `ring_toolkit.benchmark.ANOMALY_CODES` (коды `ring_toolkit.analyze`
с тем же смыслом засчитываются через `ANALYZER_ALIASES`). Невалидный ответ, исчерпанный
лимит шагов или времени считаются ошибкой и входят в сводку, а не отбрасываются.

## Метрики сводки (`summary.csv`, `summary.md`)

- `valid_rate` — доля валидных ответов;
- `regime_accuracy` — доля верных режимов (невалидный ответ — промах);
- `q_i_median_rel_error` — медиана |q_i − Q_i| / Q_i по ответам с числом;
- `q_i_within_20pct` — доля случаев с ошибкой ≤ 20 % (от всех случаев);
- `anomaly_recall` — доля ожидаемых аномалий, названных агентом (по случаям с аномалиями);
- `false_anomalies_per_case` — названные, но не ожидаемые коды;
- `mean_tokens`, `mean_time_s`.

Полный лог каждого прогона — `eval_runs/logs/<модель>__<режим>__<случай>.json`: конфиг
модели, все вызовы инструментов с выводом, переписка, ответ, оценка, токены, время.

## Изоляция эталона

Агент работает во временной копии случая без `truth.json`, папка называется `case`
(исходный `case_id` не виден). `read_file` и `list_files` не выходят за рабочую папку и не
открывают `truth.json`. `run_python` выполняется в подпроцессе с минимальным окружением
(ключи API туда не попадают) и audit-хуком, который запрещает открывать `truth.json`,
читать папки случаев и логов и запускать внешние процессы. Эталон читается только после
того, как агент закончил.

Это защита от случайного доступа, а не песочница: код через `ctypes` может обойти
audit-хук. Для недоверенных моделей запускайте стенд в контейнере, где папки случаев
монтируются только для процесса стенда.
