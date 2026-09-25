"""Прогон матрицы модель x режим x случай, логи и сводная таблица."""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .agent import run_agent
from .client import ChatClient, OpenAICompatClient
from .config import EvalConfig, ModelConfig
from .scoring import ANSWER_KEYS, expected_regime, known_codes, parse_answer, score
from .tools import MODES, REGISTRY, TRUTH_NAME, ToolContext, ToolRegistry

WORKDIR_NAME = "case"  # нейтральное имя: исходный case_id агенту не показываем


# ----------------------------------------------------------------------
# Случаи и рабочие папки
# ----------------------------------------------------------------------
def list_cases(cases_dir: str | Path) -> list[Path]:
    """Папки случаев: подкаталоги с truth.json и хотя бы одним run_*."""
    return sorted(
        p
        for p in Path(cases_dir).iterdir()
        if p.is_dir() and (p / TRUTH_NAME).is_file() and any(p.glob("run_*"))
    )


def load_truth(case_dir: Path) -> dict:
    return json.loads((case_dir / TRUTH_NAME).read_text(encoding="utf-8"))


def make_workspace(case_dir: Path, parent: Path) -> Path:
    """Копия случая без truth.json."""
    workdir = parent / WORKDIR_NAME
    shutil.copytree(case_dir, workdir, ignore=shutil.ignore_patterns(TRUTH_NAME))
    leaked = list(workdir.rglob(TRUTH_NAME))
    if leaked:  # не должно случаться; проверяем, а не надеемся
        raise RuntimeError(f"truth.json попал в рабочую папку: {leaked}")
    return workdir


# ----------------------------------------------------------------------
# Промпты
# ----------------------------------------------------------------------
_TASK = """\
The working directory holds one measurement of an all-pass microring resonator
(through port), in the ring_toolkit sweep format:
  run_000/spectrum.npz  arrays lam_nm (wavelength, nm), t_through, t_in;
                        normalised transmission is t_through / t_in
  run_000/params.json   what the experimenter knows: ring_radius (um),
                        coupling_length (um), lam0_nm, port and, if available,
                        kappa2_design — the design estimate of the power
                        coupling kappa^2 (+-25 %)
  sweep.json            sweep metadata

Determine, at 1550 nm:
  - the coupling regime: one of "overcoupled", "undercoupled", "critical",
    or "ambiguous" if the data do not determine it;
  - the intrinsic quality factor Q_i (a number);
  - which of these anomalies are present in the data (use the codes exactly):
{codes}
Note: |T| of an all-pass ring is symmetric in the self-coupling t and the
round-trip amplitude a, so a single through-port spectrum alone does not tell
over- from undercoupling. Decide only if the data give you a way to (for
example kappa2_design); otherwise answer "ambiguous" rather than guess.

When you are done, reply with ONLY a JSON object, no other text, no code fences:
{{"regime": "...", "q_i": <number or null>, "anomalies": ["CODE", ...], "confidence": <0..1>}}
"""

_MODE_HINT = {
    "naive": "You can list and read files and run Python code (numpy and scipy are available).",
    "operator": (
        "You can list files and run the ring_toolkit analysis, which returns analysis.json "
        "(loaded Q per resonance, FSR, n_g and anomaly codes of the analysis tool)."
    ),
}


def build_prompts(mode: str) -> tuple[str, str]:
    codes = "\n".join(f"    {k}: {v}" for k, v in known_codes().items())
    system = (
        "You are a careful photonics measurement analyst. Use the tools to inspect the data. "
        + _MODE_HINT.get(mode, "")
    )
    return system, _TASK.format(codes=codes)


# ----------------------------------------------------------------------
# Прогон
# ----------------------------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def run_case(
    client: ChatClient,
    model: ModelConfig,
    mode: str,
    case_dir: Path,
    cfg: EvalConfig,
    forbidden_roots: tuple[Path, ...] = (),
    registry: ToolRegistry = REGISTRY,
) -> dict:
    """Один прогон агента на одном случае; возвращает запись лога."""
    tools = registry.subset(MODES[mode])
    system, user = build_prompts(mode)
    with tempfile.TemporaryDirectory(prefix="eval_ws_") as tmp:
        workdir = make_workspace(case_dir, Path(tmp))
        ctx = ToolContext(
            workdir=workdir,
            timeout_s=cfg.run.tool_timeout_s,
            forbidden_roots=(case_dir.parent.resolve(), *forbidden_roots),
        )
        result = run_agent(
            client, tools, ctx, system, user,
            max_steps=cfg.run.max_steps,
            time_limit_s=cfg.run.time_limit_s,
            max_tool_output_chars=cfg.run.max_tool_output_chars,
        )
    # эталон читается только здесь, после того как агент закончил
    truth = load_truth(case_dir)
    answer, err = parse_answer(result.final_text)
    if answer is None and result.stop_reason != "final":
        err = f"{result.stop_reason}: {result.error or 'агент не дал финального ответа'}"
    sc = score(answer, truth, err)
    return {
        "model": model.public(),
        "mode": mode,
        "case_id": case_dir.name,
        "trap": truth["trap"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "elapsed_s": result.elapsed_s,
        "usage": result.usage,
        "tool_calls": [tc.__dict__ for tc in result.tool_calls],
        "final_text": result.final_text,
        "answer": None if answer is None else {k: getattr(answer, k) for k in ANSWER_KEYS},
        "score": sc.to_dict(),
        "truth": {
            "regime": truth["regime"],
            "regime_identifiable": truth.get("regime_identifiable", True),
            "expected_regime": expected_regime(truth),
            "q_i": truth["q_i"],
            "expected_anomalies": truth["expected_anomalies"],
        },
        "messages": result.messages,
    }


def run_benchmark(
    cfg: EvalConfig,
    cases_dir: str | Path,
    out_dir: str | Path,
    client_factory: Callable[[ModelConfig], ChatClient] = OpenAICompatClient,
    progress: Callable[[str], None] | None = print,
) -> list[dict]:
    """Все модели x режимы x случаи. Лог каждого прогона — out_dir/logs/*.json."""
    cases = list_cases(cases_dir)
    if not cases:
        raise ValueError(f"в {cases_dir} нет случаев (папок с truth.json и run_*)")
    out_dir = Path(out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    forbidden = (out_dir.resolve(),)  # в логах есть истина — агенту туда нельзя

    records = []
    for model in cfg.models:
        client = client_factory(model)
        for mode in cfg.run.modes:
            for case_dir in cases:
                rec = run_case(client, model, mode, case_dir, cfg, forbidden)
                name = f"{_slug(model.name)}__{mode}__{case_dir.name}.json"
                (logs_dir / name).write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                records.append(rec)
                if progress:
                    s = rec["score"]
                    progress(
                        f"{model.name} {mode} {case_dir.name}: "
                        f"valid={s['valid']} regime={s['regime_correct']} "
                        f"q_i_err={s['q_i_rel_error']} stop={rec['stop_reason']}"
                    )
    write_summary(records, out_dir)
    return records


# ----------------------------------------------------------------------
# Сводка
# ----------------------------------------------------------------------
SUMMARY_COLUMNS = (
    "model", "mode", "n_cases", "valid_rate", "regime_accuracy", "q_i_median_rel_error",
    "q_i_within_20pct", "anomaly_recall", "false_anomalies_per_case", "mean_tokens",
    "mean_time_s",
)


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def summarize(records: Iterable[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        groups.setdefault((r["model"]["name"], r["mode"]), []).append(r)

    rows = []
    for (model, mode), recs in sorted(groups.items()):
        scores = [r["score"] for r in recs]
        n = len(recs)
        q_errs = [s["q_i_rel_error"] for s in scores if s["q_i_rel_error"] is not None]
        recalls = [s["anomaly_recall"] for s in scores if s["anomaly_recall"] is not None]
        tokens = [r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in recs]
        rows.append(
            {
                "model": model,
                "mode": mode,
                "n_cases": n,
                "valid_rate": sum(s["valid"] for s in scores) / n,
                "regime_accuracy": sum(s["regime_correct"] for s in scores) / n,
                "q_i_median_rel_error": statistics.median(q_errs) if q_errs else None,
                # невалидные ответы и q_i = null считаются промахом
                "q_i_within_20pct": sum(e <= 0.2 for e in q_errs) / n,
                "anomaly_recall": _mean(recalls),
                "false_anomalies_per_case": sum(len(s["false_anomalies"]) for s in scores) / n,
                "mean_tokens": _mean(tokens),
                "mean_time_s": _mean([r["elapsed_s"] for r in recs]),
            }
        )
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def write_summary(records: Iterable[dict], out_dir: str | Path) -> tuple[Path, Path]:
    rows = summarize(records)
    out_dir = Path(out_dir)
    csv_path, md_path = out_dir / "summary.csv", out_dir / "summary.md"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "|" + "---|" * len(SUMMARY_COLUMNS),
        *("| " + " | ".join(_fmt(r[c]) for c in SUMMARY_COLUMNS) + " |" for r in rows),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def load_logs(out_dir: str | Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((Path(out_dir) / "logs").glob("*.json"))
    ]
