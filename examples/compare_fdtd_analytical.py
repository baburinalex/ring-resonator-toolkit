"""
Сравнение FDTD-спектра кольца с аналитической CMT-моделью.

Накладывает измеренный FDTD-спектр и аналитическую кривую на одной сетке длин
волн и печатает таблицу FOM (Q, FSR), извлечённых через ring_toolkit.analysis
из обоих источников.

Запуск в терминале (НЕ через кнопку Debug VS Code):
    python examples/compare_fdtd_analytical.py [path/to/fdtd_spectrum.npz|.csv]

Без аргумента строится синтетический "FDTD" из самой модели + шум — это лишь
демонстрация пайплайна. Подставь свои данные: либо аргументом-файлом, либо
получи спектр вживую из ring_toolkit.simulation.run_ring (см. ниже).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ring_toolkit.analysis import analyze_spectrum
from ring_toolkit.analytical_model import (
    RingModelParams,
    all_pass_transmission,
    figures_of_merit,
)

# ---------------------------------------------------------------------------
# 1. Параметры модели — ЗАМЕНИ на свои (из мод-солвера и калибровки связи).
#    n_eff0/n_g0 — из мод-солвера на lam0; loss_db_cm — оценка потерь;
#    t1 — self-coupling (подгоняется под глубину/Q твоего FDTD-резонанса).
# ---------------------------------------------------------------------------
MODEL = RingModelParams(
    ring_radius=5.0,      # мкм (как в твоём FDTD)
    coupling_length=0.0,  # мкм (racetrack -> > 0)
    n_eff0=2.40,          # из мод-солвера на 1550 нм (уточни своим значением)
    n_g0=4.15,            # из FSR ~18.4 нм твоего FDTD
    lam0_nm=1550.0,
    loss_db_cm=19.0,      # подгонка под глубину/Q резонанса
    t1=0.96,              # подгонка под глубину/Q резонанса
)


def load_fdtd_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загрузка FDTD-спектра.

    Поддерживается .npz (ключи lam_nm + t|t_norm) и текстовые .csv/.txt/.dat с двумя
    столбцами (lambda_nm, T). Необязательная строка-заголовок пропускается, длины волн
    сортируются по возрастанию.
    """
    if path.suffix == ".npz":
        d = np.load(path)
        lam = np.asarray(d["lam_nm"], dtype=float)
        t = np.asarray(d["t_norm"] if "t_norm" in d else d["t"], dtype=float)
    elif path.suffix in (".csv", ".txt", ".dat"):
        delim = "," if path.suffix == ".csv" else None
        try:
            arr = np.loadtxt(path, delimiter=delim)
        except ValueError:
            arr = np.loadtxt(path, delimiter=delim, skiprows=1)  # пропустить заголовок
        lam, t = arr[:, 0].astype(float), arr[:, 1].astype(float)
    else:
        raise ValueError(f"неизвестный формат: {path.suffix} (ожидается .npz/.csv/.txt/.dat)")
    order = np.argsort(lam)  # по возрастанию длины волны
    return lam[order], t[order]


def _synthetic_fdtd(lam_nm: np.ndarray) -> np.ndarray:
    """Заглушка вместо FDTD: спектр модели с лёгким сдвигом потерь + шум."""
    noisy = RingModelParams(**{**MODEL.model_dump(), "loss_db_cm": MODEL.loss_db_cm * 1.15})
    rng = np.random.default_rng(0)
    return np.clip(all_pass_transmission(lam_nm, noisy) + rng.normal(0, 0.004, lam_nm.size), 0, 1)


def main() -> None:
    # сетка длин волн: либо из загруженного FDTD, либо общая
    if len(sys.argv) > 1:
        lam_fdtd, t_fdtd = load_fdtd_spectrum(Path(sys.argv[1]))
        source = sys.argv[1]
    else:
        lam_fdtd = np.linspace(1500.0, 1600.0, 8000)
        t_fdtd = _synthetic_fdtd(lam_fdtd)
        source = "synthetic (no file given)"
        # Альтернатива — вживую из Lumerical:
        #   from ring_toolkit.simulation import run_ring
        #   from ring_toolkit.params import RingParams, Platform, SimParams
        #   spec = run_ring(RingParams(ring_radius=5.0, gap=0.2, wg_width=0.5,
        #                              platform=Platform.soi(),
        #                              sim=SimParams(lam_start=1500.0, lam_stop=1600.0)))
        #   lam_fdtd, t_fdtd = spec.lam_nm, spec.t_norm

    # аналитическая кривая на той же сетке
    t_model = all_pass_transmission(lam_fdtd, MODEL)

    # FOM из обоих спектров через общий анализатор
    fdtd = analyze_spectrum(lam_fdtd, t_fdtd)
    fom = figures_of_merit(MODEL)

    print(f"FDTD source: {source}\n")
    print(f"{'metric':<12}{'FDTD':>14}{'analytic':>14}")
    print(f"{'FSR (nm)':<12}{fdtd.mean_fsr_nm:>14.3f}{fom.fsr_nm:>14.3f}")
    fdtd_q = fdtd.resonances[len(fdtd.resonances) // 2].q if fdtd.resonances else float("nan")
    print(f"{'Q (loaded)':<12}{fdtd_q:>14.0f}{fom.q_loaded:>14.0f}")
    print(f"\nmodel FOM:\n{fom.report()}")

    # оверлей
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib не установлен — график пропущен)")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lam_fdtd, t_fdtd, lw=0.8, alpha=0.7, label="FDTD")
    ax.plot(lam_fdtd, t_model, lw=1.2, label="analytic (CMT)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalized transmission")
    ax.set_title("Ring resonator: FDTD vs analytical model")
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).parent / "results" / "fdtd_vs_analytical.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nоверлей сохранён: {out}")


if __name__ == "__main__":
    main()
