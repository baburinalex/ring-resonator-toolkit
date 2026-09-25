"""CLI: анализ папки свипа -> analysis.json.

Запуск из корня репозитория:
    python -m ring_toolkit.analyze sweeps/<имя_свипа>

Модуль ничего не интерпретирует физически: он вызывает analyze_spectrum,
проверяет надёжность чисел и выносит всё подозрительное в список аномалий.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path

import numpy as np

from .analysis import SpectrumAnalysis, analyze_spectrum, estimate_q, find_resonances
from .sweep_io import list_runs, load_run

MIN_POINTS_PER_FWHM = 5.0  # меньше -> Q считается ненадёжной
FSR_SPREAD_TOL = 0.05  # (max-min)/mean по FSR; больше -> вероятен пропуск резонанса
N_G_MIN_PLAUSIBLE = 1.5  # нижняя граница n_g для оценки ожидаемого числа резонансов

THROUGH_PORT_NOTE = (
    "Q — нагруженная. По одному through-порту режимы пере- и недосвязи "
    "неразличимы: kappa^2 и собственные потери однозначно не извлекаются."
)


def _default(func, name: str) -> float:
    """Значение параметра по умолчанию — читаем из сигнатуры, а не дублируем."""
    return float(inspect.signature(func).parameters[name].default)


def _num(x: float) -> float | None:
    """NaN/inf -> None, чтобы JSON оставался валидным."""
    return float(x) if math.isfinite(x) else None


def _find_key(obj, key: str):
    """Рекурсивный поиск ключа в params.json (модель параметров может быть вложенной)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def check_analysis(
    analysis: SpectrumAnalysis,
    lam_nm: np.ndarray,
    win_nm: float,
    min_sep_nm: float,
    round_trip_um: float | None = None,
) -> dict:
    """Превращает SpectrumAnalysis в JSON-совместимый словарь с проверками."""
    # шаг — по уникальным λ: повторные отсчёты на одной длине волны дают нулевой
    # медианный шаг и бесконечное число точек на FWHM
    unique_lam = np.unique(np.asarray(lam_nm, dtype=float))
    step_nm = float(np.median(np.diff(unique_lam))) if unique_lam.size > 1 else float("nan")
    anomalies: list[dict] = []
    rows: list[dict] = []

    for r in analysis.resonances:
        if math.isfinite(r.q) and r.q > 0 and step_nm > 0:
            points = (r.lambda0_nm / r.q) / step_nm
            reliable = math.isfinite(points) and points >= MIN_POINTS_PER_FWHM
        else:
            points, reliable = float("nan"), False
        rows.append(
            {
                "lambda0_nm": float(r.lambda0_nm),
                "q_loaded": _num(r.q),
                "depth": float(r.depth),
                "points_per_fwhm": _num(points),
                "q_reliable": bool(reliable),
            }
        )

    n_failed = sum(1 for row in rows if row["q_loaded"] is None)
    n_under = sum(1 for row in rows if row["q_loaded"] is not None and not row["q_reliable"])

    if not rows:
        anomalies.append({"code": "NO_RESONANCES", "message": "Резонансы не найдены."})
    if len(rows) == 1:
        anomalies.append(
            {
                "code": "FSR_UNDETERMINED",
                "message": (
                    "Найден один резонанс: FSR не определён, "
                    "проверки окна и min_sep_nm невозможны."
                ),
            }
        )
    if round_trip_um:
        span_nm = float(np.max(lam_nm) - np.min(lam_nm))
        lam_c = float(np.mean(lam_nm))
        expected_min = span_nm * N_G_MIN_PLAUSIBLE * round_trip_um * 1e3 / lam_c**2
        if len(rows) < 0.5 * expected_min:
            anomalies.append(
                {
                    "code": "RESONANCE_COUNT_LOW",
                    "message": (
                        f"Найдено {len(rows)} резонансов, а по геометрии (обход "
                        f"{round_trip_um:.4g} мкм, n_g >= {N_G_MIN_PLAUSIBLE}) в диапазоне "
                        f"{span_nm:.3g} нм ожидается не меньше {expected_min:.1f}."
                    ),
                }
            )
    if n_failed:
        anomalies.append(
            {
                "code": "Q_FAILED",
                "message": f"Q не определилась для {n_failed} из {len(rows)} резонансов.",
            }
        )
    if n_under:
        anomalies.append(
            {
                "code": "UNDERSAMPLED_FWHM",
                "message": (
                    f"{n_under} резонансов: меньше {MIN_POINTS_PER_FWHM:g} точек сетки "
                    f"на FWHM (шаг {step_nm:.4g} нм). Q ненадёжна."
                ),
            }
        )

    fsr = analysis.mean_fsr_nm
    if math.isfinite(fsr):
        if win_nm >= fsr:
            anomalies.append(
                {
                    "code": "WINDOW_GE_FSR",
                    "message": (
                        f"Окно estimate_q ±{win_nm:g} нм >= FSR {fsr:.3g} нм: "
                        "в окно попадают соседние резонансы."
                    ),
                }
            )
        if fsr < 2 * min_sep_nm:
            anomalies.append(
                {
                    "code": "MIN_SEP_NEAR_FSR",
                    "message": (
                        f"FSR {fsr:.3g} нм < 2*min_sep_nm ({min_sep_nm:g} нм): "
                        "find_resonances мог слить или пропустить резонансы."
                    ),
                }
            )
        spread = (max(analysis.fsr_nm) - min(analysis.fsr_nm)) / fsr
        if spread > FSR_SPREAD_TOL:
            anomalies.append(
                {
                    "code": "FSR_NONUNIFORM",
                    "message": (
                        f"Разброс FSR {spread:.1%} > {FSR_SPREAD_TOL:.0%}: "
                        "пропущенный резонанс или второе семейство мод."
                    ),
                }
            )

    reliable_q = [row["q_loaded"] for row in rows if row["q_reliable"]]
    return {
        "n_resonances": len(rows),
        "n_q_failed": n_failed,
        "n_q_unreliable": n_under,
        "mean_fsr_nm": _num(fsr),
        "n_g": _num(analysis.n_g),
        "mean_q_loaded_reliable": _num(float(np.mean(reliable_q))) if reliable_q else None,
        "grid_step_nm": step_nm,
        "resonances": rows,
        "anomalies": anomalies,
    }


def analyze_sweep(sweep_dir: str | Path, min_depth: float = 0.15) -> dict:
    sweep_dir = Path(sweep_dir)
    win_nm = _default(estimate_q, "win_nm")
    min_sep_nm = _default(find_resonances, "min_sep_nm")

    runs = []
    for run_dir in list_runs(sweep_dir):
        spectrum, params = load_run(run_dir)
        radius = _find_key(params, "radius_um") or _find_key(params, "ring_radius")
        coupling = _find_key(params, "coupling_length_um") or _find_key(params, "coupling_length")
        radius = float(radius) if radius is not None else None
        coupling = float(coupling) if coupling is not None else 0.0
        round_trip = 2.0 * math.pi * radius + 2.0 * coupling if radius else None
        analysis = analyze_spectrum(
            spectrum.lam_nm,
            np.real(spectrum.t_norm),
            min_depth=min_depth,
            radius_um=radius,
            coupling_length_um=coupling,
        )
        result = check_analysis(analysis, spectrum.lam_nm, win_nm, min_sep_nm, round_trip)
        runs.append({"run": run_dir.name, "params": params, **result})

    return {
        "sweep": sweep_dir.name,
        "settings": {"min_depth": min_depth, "win_nm": win_nm, "min_sep_nm": min_sep_nm},
        "note": THROUGH_PORT_NOTE,
        "n_runs": len(runs),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Анализ папки свипа -> analysis.json")
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--min-depth", type=float, default=0.15)
    args = parser.parse_args(argv)

    if not list_runs(args.sweep_dir):
        print(f"В {args.sweep_dir} нет папок run_*")
        return 2

    report = analyze_sweep(args.sweep_dir, min_depth=args.min_depth)
    out = args.sweep_dir / "analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for run in report["runs"]:
        codes = ", ".join(a["code"] for a in run["anomalies"]) or "ok"
        print(f"{run['run']}: резонансов {run['n_resonances']}, аномалии: {codes}")
    print(f"Записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
