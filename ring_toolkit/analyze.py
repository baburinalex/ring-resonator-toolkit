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
from scipy.signal import find_peaks

from .analysis import SpectrumAnalysis, analyze_spectrum, estimate_q, find_resonances
from .sweep_io import list_runs, load_run

MIN_POINTS_PER_FWHM = 5.0  # меньше -> Q считается ненадёжной
FSR_SPREAD_TOL = 0.05  # (max-min)/mean по FSR; больше -> вероятен пропуск резонанса
N_G_MIN_PLAUSIBLE = 1.5  # нижняя граница n_g для оценки ожидаемого числа резонансов

# Обнаружение формы (только коды аномалий, без подгонки моделей дублета или фона)
SPLIT_WIN_FWHM = 3.0  # дублет ищем в окне +-3 FWHM вокруг резонанса
SPLIT_MIN_REL_PROMINENCE = 0.1  # выступ между минимумами >= 10 % глубины резонанса...
NOISE_SIGMAS = 4.0  # ...и >= 4 шумовых сигм
SPLIT_MIN_REL_DEPTH = 0.3  # оба минимума дублета глубже 30 % глубины резонанса
NARROW_DIP_FWHM = 3.0  # провал уже 3 FWHM — резонанс (свой или чужой), шире — фон
NARROW_DIP_MIN_DEPTH = 0.02  # мельче не маскируем: это уже шум или рябь
BASELINE_MASK_FWHM = 5.0  # точки ближе 5 FWHM к узкому провалу в базовую линию не входят
BASELINE_MIN_POINTS = 20
TILT_TOL = 0.05  # относительный перепад линейного тренда базы по диапазону
RIPPLE_TOL = 0.01  # относительное СКО базы после вычета тренда (за вычетом шума)

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


def _unique_mean(lam_nm: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сортировка по λ и усреднение повторных отсчётов на одной длине волны."""
    lam_u, inv = np.unique(np.asarray(lam_nm, dtype=float), return_inverse=True)
    t_u = np.bincount(inv, weights=np.asarray(t, dtype=float)) / np.bincount(inv)
    return lam_u, t_u


def _noise_sigma(t: np.ndarray) -> float:
    """Шум по разностям соседних точек (MAD): гладкие провалы и рябь почти не влияют."""
    d = np.diff(t)
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


def detect_shape_anomalies(
    analysis: SpectrumAnalysis,
    lam_nm: np.ndarray,
    t: np.ndarray,
    min_sep_nm: float,
) -> list[dict]:
    """Дублеты и паразитный фон: только обнаружение, без подгонки.

    RESONANCE_SPLITTING — у резонанса в окне +-3 FWHM два минимума глубже 30 %
    его глубины, разделённых выступом не меньше 10 % глубины и 4 шумовых сигм.
    BASELINE_TILT / BASELINE_RIPPLE — по точкам дальше 5 FWHM от любого узкого
    провала (ширина не больше 3 FWHM самой узкой линии — так в маску попадают
    и резонансы, которые find_resonances слил или пропустил, а широкая рябь нет):
    линейный тренд даёт перепад больше TILT_TOL, а СКО остатка сверх шума —
    больше RIPPLE_TOL от уровня базы.
    """
    lam, tt = _unique_mean(lam_nm, t)
    if lam.size < 3:
        return []
    sigma = _noise_sigma(tt)
    fsr = analysis.mean_fsr_nm
    anomalies: list[dict] = []

    # Опорная ширина линии — самая узкая из найденных: find_resonances может
    # принять минимумы широкой ряби за «резонансы» с низкой Q, и медиана по ним
    # раздула бы маску на весь спектр. Такие широкие «резонансы» дальше не
    # считаются ни резонансами (для дублетов), ни точками маски.
    widths = {
        id(r): r.lambda0_nm / r.q for r in analysis.resonances if math.isfinite(r.q) and r.q > 0
    }
    if not widths:
        return anomalies  # без ширины линии не отличить резонансы от ряби
    fwhm = min(widths.values())
    narrow = [
        r for r in analysis.resonances if widths.get(id(r), math.inf) <= NARROW_DIP_FWHM * fwhm
    ]

    split_at = []
    for r in narrow:
        half = SPLIT_WIN_FWHM * r.lambda0_nm / r.q
        if math.isfinite(fsr):
            half = min(half, 0.4 * fsr)
        m = np.abs(lam - r.lambda0_nm) <= half
        if m.sum() < 7:
            continue
        seg = tt[m]
        min_prom = max(SPLIT_MIN_REL_PROMINENCE * r.depth, NOISE_SIGMAS * sigma)
        minima, _ = find_peaks(-seg, prominence=min_prom)
        deep = minima[seg.max() - seg[minima] >= SPLIT_MIN_REL_DEPTH * r.depth]
        if len(deep) >= 2:
            split_at.append(r.lambda0_nm)
    if split_at:
        anomalies.append(
            {
                "code": "RESONANCE_SPLITTING",
                "message": (
                    f"{len(split_at)} из {len(narrow)} резонансов расщеплены "
                    f"в дублет (например, у {split_at[0]:.3f} нм): вероятно обратное "
                    "рассеяние. Q по ширине всего провала занижена."
                ),
            }
        )

    step = float(np.median(np.diff(lam)))
    dips, _ = find_peaks(
        -tt,
        prominence=max(NARROW_DIP_MIN_DEPTH, NOISE_SIGMAS * sigma),
        width=(None, NARROW_DIP_FWHM * fwhm / step),
    )
    keep = np.ones(lam.size, dtype=bool)
    for center in [*lam[dips], *(r.lambda0_nm for r in narrow)]:
        keep &= np.abs(lam - center) > BASELINE_MASK_FWHM * fwhm
    span = float(lam[-1] - lam[0])
    base_lam, base_t = lam[keep], tt[keep]
    if base_lam.size < BASELINE_MIN_POINTS or np.ptp(base_lam) < 0.3 * span:
        return anomalies

    x = base_lam - base_lam.mean()
    slope, level = np.polyfit(x, base_t, 1)
    if level <= 0:
        return anomalies
    tilt = abs(slope) * span / level
    resid = base_t - (slope * x + level)
    ripple = math.sqrt(max(float(np.mean(resid**2)) - sigma**2, 0.0)) / level
    if tilt > TILT_TOL:
        anomalies.append(
            {
                "code": "BASELINE_TILT",
                "message": (
                    f"Базовая линия наклонена: перепад {tilt:.0%} по диапазону "
                    f"{span:.3g} нм (порог {TILT_TOL:.0%}). Спектр не нормирован на опорный."
                ),
            }
        )
    if ripple > RIPPLE_TOL:
        anomalies.append(
            {
                "code": "BASELINE_RIPPLE",
                "message": (
                    f"Рябь базовой линии: СКО {ripple:.1%} от уровня сверх шума "
                    f"(порог {RIPPLE_TOL:.0%}) — паразитный Фабри-Перо или неучтённые провалы."
                ),
            }
        )
    return anomalies


def check_analysis(
    analysis: SpectrumAnalysis,
    lam_nm: np.ndarray,
    win_nm: float,
    min_sep_nm: float,
    round_trip_um: float | None = None,
    t: np.ndarray | None = None,
) -> dict:
    """Превращает SpectrumAnalysis в JSON-совместимый словарь с проверками.

    Если передан спектр ``t``, дополнительно ищутся дублеты и паразитный фон.
    """
    step_nm = float(np.median(np.abs(np.diff(lam_nm))))
    anomalies: list[dict] = []
    rows: list[dict] = []

    for r in analysis.resonances:
        if math.isfinite(r.q) and r.q > 0:
            points = (r.lambda0_nm / r.q) / step_nm
            reliable = points >= MIN_POINTS_PER_FWHM
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

    if t is not None:
        anomalies.extend(detect_shape_anomalies(analysis, lam_nm, t, min_sep_nm))

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
        t_norm = np.real(spectrum.t_norm)
        analysis = analyze_spectrum(
            spectrum.lam_nm,
            t_norm,
            min_depth=min_depth,
            radius_um=radius,
            coupling_length_um=coupling,
        )
        result = check_analysis(
            analysis, spectrum.lam_nm, win_nm, min_sep_nm, round_trip, t=t_norm
        )
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
