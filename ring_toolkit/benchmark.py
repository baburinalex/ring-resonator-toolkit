"""Генератор размеченных спектров кольцевых резонаторов с ловушками.

Каждый случай — папка в формате ``ring_toolkit.sweep_io`` (``run_000/`` со
``spectrum.npz`` и ``params.json``, плюс ``sweep.json``) и ``truth.json`` с
истинными значениями. Базовый спектр считает ``ring_toolkit.analytical_model``
(all-pass кольцо); ловушка — одно искажение поверх него.

Запуск из корня репозитория:
    python -m ring_toolkit.benchmark sweeps/benchmark --seed 0
    python -m ring_toolkit.benchmark sweeps/benchmark --seed 1 --traps doublet small_fsr -n 3

Что видит анализируемый инструмент, а что — нет:
    - ``params.json`` содержит только то, что знает экспериментатор: радиус,
      длину участка связи, опорную длину волны и проектную оценку kappa^2
      (``kappa2_design``, ошибка до +-25 %). t1 и потери туда не пишутся —
      иначе Q_i и режим связи читались бы из параметров, а не из спектра.
    - имя папки случая и ``sweep.json`` тип ловушки не раскрывают.
    - ``truth.json`` лежит рядом и при проверке инструмента должен быть убран.

Зачем ``kappa2_design``: |T| all-pass кольца симметричен по t <-> a, поэтому по
одному through-порту пере- и недосвязь неразличимы. Режимы в генераторе
разнесены (kappa^2 / (1 - a^2) >= 2 или <= 0.5), и проектной оценки с ошибкой
+-25 % достаточно, чтобы выбрать правильную ветку.

Без подсказки (``--no-design-hint``, ``design_hint=False``) ``kappa2_design`` в
params.json не пишется, а в truth.json ``regime_identifiable = false`` — кроме
ловушки ``kappa_dispersion``: там связь меняется по диапазону, а потери нет, и
ветка определяется по тому, какой из двух параметров «плывёт». Спектр при этом
тот же, что и с подсказкой (одинаковый seed — одинаковые массивы).

Режим связи определяется отношением rho = kappa^2 / (1 - a^2) на 1550 нм:
    rho > 1.1 — пересвязь, rho < 0.9 — недосвязь, иначе — критическая связь.

Все случайные параметры выводятся из seed; одинаковые (trap, seed) дают
побайтно одинаковые массивы и truth.json.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytical_model import RingModelParams, all_pass_transmission, figures_of_merit
from .simulation import RingSpectrum
from .sweep_io import save_run

LAM0_NM = 1550.0
# SiN-подобный волновод, как в examples/make_demo_sweep.py
WAVEGUIDE = {"n_eff0": 1.8, "n_g0": 2.1, "lam0_nm": LAM0_NM}
NOISE_STD = 0.002  # шум детектора в единицах нормированного пропускания

TRAPS: tuple[str, ...] = (
    "overcoupled",
    "undercoupled",
    "critical",
    "undersampled",
    "duplicate_samples",
    "doublet",
    "baseline",
    "kappa_dispersion",
    "two_families",
    "small_fsr",
)

REGIMES: tuple[str, ...] = ("overcoupled", "undercoupled", "critical")

# Словарь кодов аномалий, которые должен назвать анализирующий инструмент.
ANOMALY_CODES: dict[str, str] = {
    "UNDERSAMPLED": "шаг сетки сравним с шириной линии: Q по спектру ненадёжна",
    "DUPLICATE_SAMPLES": "несколько отсчётов на одной длине волны",
    "RESONANCE_SPLITTING": "резонансы расщеплены в дублет (обратное рассеяние)",
    "BASELINE_RIPPLE": "паразитная рябь Фабри-Перо на базовой линии",
    "BASELINE_TILT": "наклон базовой линии по длине волны",
    "COUPLING_DISPERSION": "коэффициент связи kappa^2 заметно меняется по диапазону",
    "MULTIPLE_MODE_FAMILIES": "в спектре два семейства мод с разным FSR",
    "SMALL_FSR": "FSR меньше 1 нм: соседние резонансы близко, окна анализа велики",
}

# Коды ring_toolkit.analyze с тем же смыслом (для нормализации ответов).
ANALYZER_ALIASES: dict[str, str] = {
    "UNDERSAMPLED_FWHM": "UNDERSAMPLED",
    "WINDOW_GE_FSR": "SMALL_FSR",
    "MIN_SEP_NEAR_FSR": "SMALL_FSR",
}

_EXPECTED: dict[str, list[str]] = {
    "overcoupled": [],
    "undercoupled": [],
    "critical": [],
    "undersampled": ["UNDERSAMPLED"],
    "duplicate_samples": ["DUPLICATE_SAMPLES"],
    "doublet": ["RESONANCE_SPLITTING"],
    "baseline": ["BASELINE_RIPPLE", "BASELINE_TILT"],
    "kappa_dispersion": ["COUPLING_DISPERSION"],
    "two_families": ["MULTIPLE_MODE_FAMILIES"],
    "small_fsr": ["SMALL_FSR"],
}


@dataclass
class Case:
    """Один размеченный случай: спектр, открытые параметры и истина."""

    case_id: str
    trap: str
    spectrum: RingSpectrum
    params: dict
    truth: dict


# ----------------------------------------------------------------------
# Физика
# ----------------------------------------------------------------------
def classify_regime(rho: float) -> str:
    """Режим связи по rho = kappa^2 / (1 - a^2)."""
    if rho > 1.1:
        return "overcoupled"
    if rho < 0.9:
        return "undercoupled"
    return "critical"


def through_field(
    lam_nm: np.ndarray, p: RingModelParams, t1: np.ndarray | float | None = None
) -> np.ndarray:
    """Комплексная амплитуда through-порта all-pass кольца.

    |through_field|^2 совпадает с analytical_model.all_pass_transmission;
    ``t1`` можно задать массивом (дисперсия связи по длине волны).
    """
    t = p.t1 if t1 is None else np.asarray(t1, dtype=float)
    ea = p.amplitude_a * np.exp(1j * p.phase(lam_nm))
    return (t - ea) / (1.0 - t * ea)


def _ring(rng: np.random.Generator, radius: float, regime: str, **wg) -> RingModelParams:
    """Кольцо с потерями из seed и t1, дающим нужный режим связи."""
    p = RingModelParams(ring_radius=radius, t1=0.5, loss_db_cm=rng.uniform(2.0, 4.0),
                        **(wg or WAVEGUIDE))
    rho = {
        "overcoupled": rng.uniform(2.0, 4.0),
        "undercoupled": rng.uniform(0.25, 0.5),
        "critical": rng.uniform(0.97, 1.03),
    }[regime]
    kappa2 = rho * (1.0 - p.amplitude_a**2)
    return p.model_copy(update={"t1": math.sqrt(1.0 - kappa2)})


def _grid(center: float, span: float, step: float) -> np.ndarray:
    n = int(round(span / step))
    return center - span / 2 + step * np.arange(n + 1)


def _fwhm_nm(p: RingModelParams) -> float:
    return LAM0_NM / figures_of_merit(p).q_loaded


# ----------------------------------------------------------------------
# Ловушки: каждая возвращает (lam, T, trap_params, notes) для кольца p
# ----------------------------------------------------------------------
def _clean(rng, p, fsr, span_fsr=3.2, pts_per_fwhm=None):
    step = _fwhm_nm(p) / (pts_per_fwhm or rng.uniform(15, 25))
    lam = _grid(LAM0_NM, span_fsr * fsr, step)
    return lam, np.asarray(all_pass_transmission(lam, p)), step


def _trap_regime(rng, p, fsr):
    lam, t, step = _clean(rng, p, fsr)
    return lam, t, {"step_nm": step}, "Чистый спектр; ловушка — только режим связи."


def _trap_undersampled(rng, p, fsr):
    ratio = rng.uniform(0.8, 1.2)
    lam, t, step = _clean(rng, p, fsr, pts_per_fwhm=1.0 / ratio)
    notes = f"Шаг сетки {ratio:.2f} FWHM: провал попадает на 1-2 отсчёта, Q по ширине занижена."
    return lam, t, {"step_nm": step, "step_over_fwhm": ratio}, notes


def _trap_duplicates(rng, p, fsr):
    lam, t, step = _clean(rng, p, fsr, pts_per_fwhm=rng.uniform(15, 20))
    k = int(rng.integers(2, 4))
    notes = (
        f"Каждая длина волны записана {k} раза с независимым шумом "
        "(квантование показаний лазера): медианный шаг сетки равен нулю."
    )
    return np.repeat(lam, k), np.repeat(t, k), {"step_nm": step, "repeats": k}, notes


def _trap_doublet(rng, p, fsr):
    lam, _, step = _clean(rng, p, fsr)
    split_fwhm = rng.uniform(1.5, 3.0)  # расстояние между пиками дублета, в FWHM
    dphi = math.pi * split_fwhm * _fwhm_nm(p) / fsr  # половина расщепления по фазе
    # CW/CCW-моды, связанные обратным рассеянием: стоячие моды сдвинуты на +-dphi,
    # каждая связана с шиной вдвое слабее -> среднее двух сдвинутых амплитуд.
    ea = p.amplitude_a * np.exp(1j * p.phase(lam))
    shifted = [(p.t1 - ea * np.exp(1j * s)) / (1.0 - p.t1 * ea * np.exp(1j * s))
               for s in (dphi, -dphi)]
    t = np.abs(0.5 * (shifted[0] + shifted[1])) ** 2
    notes = (
        f"Обратное рассеяние расщепляет каждый резонанс на дублет ({split_fwhm:.2f} FWHM "
        "между компонентами). Q_i, Q_c — для отдельной моды; ширина всего провала их занижает."
    )
    return lam, t, {"step_nm": step, "split_over_fwhm": split_fwhm}, notes


def _trap_baseline(rng, p, fsr):
    lam, t, step = _clean(rng, p, fsr)
    x = (lam - LAM0_NM) / (lam.max() - lam.min())  # -0.5 .. 0.5
    tilt = rng.uniform(0.15, 0.35) * rng.choice([-1.0, 1.0])  # относительный перепад
    ripple = rng.uniform(0.04, 0.10)
    period = fsr * rng.uniform(0.15, 0.4)
    phase0 = rng.uniform(0, 2 * math.pi)
    base = (1.0 + tilt * x) * (1.0 + ripple * np.cos(2 * math.pi * lam / period + phase0))
    base *= rng.uniform(0.85, 0.95) / base.max()
    notes = (
        f"Базовая линия: наклон {tilt:+.0%} по диапазону и рябь Фабри-Перо "
        f"+-{ripple:.0%} с периодом {period:.3f} нм; спектр не нормирован на неё."
    )
    params = {"step_nm": step, "tilt": tilt, "ripple": ripple, "ripple_period_nm": period}
    return lam, t * base, params, notes


def _trap_kappa_dispersion(rng, p, fsr):
    lam, _, step = _clean(rng, p, fsr, span_fsr=6.0)
    kappa2_0 = 1.0 - p.t1**2
    slope = rng.uniform(0.3, 0.6) * rng.choice([-1.0, 1.0])  # доля изменения на полдиапазона
    half = (lam.max() - lam.min()) / 2
    kappa2 = kappa2_0 * (1.0 + slope * (lam - LAM0_NM) / half)
    t = np.abs(through_field(lam, p, np.sqrt(1.0 - kappa2))) ** 2
    rho_edges = kappa2[[0, -1]] / (1.0 - p.amplitude_a**2)
    edges = [classify_regime(r) for r in rho_edges]
    notes = (
        f"kappa^2 меняется линейно на {slope:+.0%} от центра к краю диапазона; "
        f"режим на краях: {edges[0]} / {edges[1]}. Истина — на 1550 нм."
    )
    params = {"step_nm": step, "kappa2_rel_slope_per_half_span": slope,
              "kappa2_edges": kappa2[[0, -1]].tolist()}
    return lam, t, params, notes


def _trap_two_families(rng, p, fsr):
    lam, _, step = _clean(rng, p, fsr, span_fsr=4.0)
    wg2 = {"n_eff0": rng.uniform(1.55, 1.7), "n_g0": rng.uniform(2.25, 2.4), "lam0_nm": LAM0_NM}
    p2 = _ring(rng, p.ring_radius, str(rng.choice(["overcoupled", "undercoupled"])), **wg2)
    p2 = p2.model_copy(update={"loss_db_cm": p.loss_db_cm * rng.uniform(1.5, 2.5)})
    t = np.abs(through_field(lam, p) * through_field(lam, p2)) ** 2
    fsr2 = figures_of_merit(p2).fsr_nm
    notes = (
        f"Второе семейство мод (n_g = {p2.n_g0:.3f}, FSR {fsr2:.3f} нм против {fsr:.3f} нм) "
        "в той же шине. Истина относится к основному семейству."
    )
    params = {"step_nm": step, "family2": p2.model_dump(), "family2_fsr_nm": fsr2}
    return lam, t, params, notes


def _trap_small_fsr(rng, p, fsr):
    lam, t, step = _clean(rng, p, fsr, span_fsr=12.0)
    notes = (
        f"Большое кольцо, FSR {fsr:.3f} нм < 1 нм: меньше окон estimate_q и "
        "2*min_sep_nm с настройками по умолчанию."
    )
    return lam, t, {"step_nm": step}, notes


_TRAP_FUNCS = {
    "overcoupled": _trap_regime,
    "undercoupled": _trap_regime,
    "critical": _trap_regime,
    "undersampled": _trap_undersampled,
    "duplicate_samples": _trap_duplicates,
    "doublet": _trap_doublet,
    "baseline": _trap_baseline,
    "kappa_dispersion": _trap_kappa_dispersion,
    "two_families": _trap_two_families,
    "small_fsr": _trap_small_fsr,
}


# ----------------------------------------------------------------------
# Сборка случая
# ----------------------------------------------------------------------
def is_regime_identifiable(trap: str, design_hint: bool) -> bool:
    """Можно ли по данным случая отличить пере- от недосвязи."""
    return design_hint or trap == "kappa_dispersion"


def make_case(
    trap: str, seed: int, case_id: str | None = None, design_hint: bool = True
) -> Case:
    """Строит случай с ловушкой ``trap``; всё случайное выводится из ``seed``.

    ``design_hint=False`` убирает ``kappa2_design`` из params.json; спектр не меняется.
    """
    if trap not in _TRAP_FUNCS:
        raise ValueError(f"неизвестная ловушка {trap!r}; доступны: {', '.join(TRAPS)}")
    rng = np.random.default_rng([seed, TRAPS.index(trap)])

    regime = trap if trap in REGIMES else str(rng.choice(["overcoupled", "undercoupled"]))
    radius = rng.uniform(220.0, 400.0) if trap == "small_fsr" else rng.uniform(15.0, 40.0)
    p = _ring(rng, radius, regime)
    fom = figures_of_merit(p)
    kappa2 = 1.0 - p.t1**2
    rho = kappa2 / (1.0 - p.amplitude_a**2)
    assert classify_regime(rho) == regime

    lam, t, trap_params, notes = _TRAP_FUNCS[trap](rng, p, fom.fsr_nm)
    t = np.clip(t + rng.normal(0.0, NOISE_STD, t.shape), 0.0, 1.0)

    case_id = case_id or f"{trap}_s{seed}"
    kappa2_design = kappa2 * float(np.exp(rng.uniform(-1.0, 1.0) * math.log(1.25)))
    params = {
        "ring_radius": p.ring_radius,
        "coupling_length": p.coupling_length,
        "lam0_nm": LAM0_NM,
        "port": "through",
    }
    if design_hint:
        params["kappa2_design"] = kappa2_design
    truth = {
        "case_id": case_id,
        "trap": trap,
        "regime": regime,
        "q_i": fom.q_intrinsic,
        "q_c": fom.q_coupling,
        "kappa2_1550": kappa2,
        "fsr_nm": fom.fsr_nm,
        "expected_anomalies": list(_EXPECTED[trap]),
        "notes": notes,
        "regime_identifiable": is_regime_identifiable(trap, design_hint),
        "design_hint": design_hint,
        "seed": seed,
        "q_loaded": fom.q_loaded,
        "generator": {
            "model": p.model_dump(),
            "rho": rho,
            "noise_std": NOISE_STD,
            "trap_params": trap_params,
        },
    }
    spectrum = RingSpectrum(lam_nm=lam, t_through=t, t_in=np.ones_like(lam))
    return Case(case_id, trap, spectrum, params, truth)


def write_case(case: Case, out_dir: str | Path) -> Path:
    """Пишет случай в ``out_dir/<case_id>/``: run_000/, sweep.json, truth.json."""
    case_dir = Path(out_dir) / case.case_id
    save_run(case_dir / "run_000", case.spectrum, case.params)
    meta = {"swept": None, "source": "ring_toolkit.benchmark", "n_runs": 1}
    (case_dir / "sweep.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (case_dir / "truth.json").write_text(
        json.dumps(case.truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return case_dir


def generate_benchmark(
    out_dir: str | Path,
    seed: int = 0,
    traps: tuple[str, ...] | list[str] = TRAPS,
    n_per_trap: int = 1,
    design_hint: bool = True,
) -> list[Path]:
    """Набор случаев с нейтральными именами ``case_000``... в перемешанном порядке.

    В ``out_dir/manifest.json`` — соответствие case_id -> ловушка, seed.
    """
    jobs = [(trap, seed * 1000 + k) for trap in traps for k in range(n_per_trap)]
    order = np.random.default_rng(seed).permutation(len(jobs))
    out_dir = Path(out_dir)
    manifest, paths = [], []
    for i, j in enumerate(order):
        trap, case_seed = jobs[j]
        case = make_case(trap, case_seed, case_id=f"case_{i:03d}", design_hint=design_hint)
        paths.append(write_case(case, out_dir))
        manifest.append({"case_id": case.case_id, "trap": trap, "seed": case_seed})
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {"seed": seed, "design_hint": design_hint, "cases": manifest},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Генератор размеченных спектров с ловушками")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traps", nargs="+", choices=TRAPS, default=list(TRAPS))
    parser.add_argument("-n", "--n-per-trap", type=int, default=1)
    parser.add_argument(
        "--no-design-hint",
        dest="design_hint",
        action="store_false",
        help="не писать kappa2_design в params.json (режим связи становится невосстановимым)",
    )
    args = parser.parse_args(argv)

    paths = generate_benchmark(
        args.out_dir, args.seed, args.traps, args.n_per_trap, design_hint=args.design_hint
    )
    for path in paths:
        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        print(f"{path.name}: {truth['trap']:<18} {truth['regime']:<13} "
              f"Q_i={truth['q_i']:.3g} Q_c={truth['q_c']:.3g} FSR={truth['fsr_nm']:.3f} нм")
    print(f"Записано случаев: {len(paths)} в {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
