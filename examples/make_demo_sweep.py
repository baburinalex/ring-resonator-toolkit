"""Генерирует демонстрационные свипы аналитической моделью (Lumerical не нужен).

Запуск из корня репозитория:
    python examples/make_demo_sweep.py

Создаёт:
    sweeps/demo_t1/         свип по t1 через критическую связь; последний прогон
                            посчитан на заведомо грубой сетке (ловушка)
    sweeps/demo_small_fsr/  большое кольцо с FSR < 1 нм: дефолтные окна анализа
                            toolkit для него не подходят (ловушка)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ring_toolkit.analytical_model import RingModelParams, all_pass_transmission
from ring_toolkit.simulation import RingSpectrum
from ring_toolkit.sweep_io import save_run

ROOT = Path("sweeps")

# SiN-подобный волновод. Потери завышены намеренно, чтобы критическая связь
# попала в удобный диапазон t1 и линии не были слишком узкими для демо.
BASE = {"n_eff0": 1.8, "n_g0": 2.1, "lam0_nm": 1550.0, "loss_db_cm": 2.0}

Grid = tuple[float, float, float]  # lam_start, lam_stop, step (нм)


def _spectrum(p: RingModelParams, grid: Grid) -> RingSpectrum:
    lam_start, lam_stop, step_nm = grid
    lam = np.arange(lam_start, lam_stop + step_nm / 2, step_nm)
    t = np.asarray(all_pass_transmission(lam, p))
    if np.iscomplexobj(t):  # на случай, если модель отдаёт амплитуду, а не мощность
        t = np.abs(t) ** 2
    return RingSpectrum(lam_nm=lam, t_through=t, t_in=np.ones_like(lam))


def _write_sweep(name: str, swept: str, runs: list[tuple[RingModelParams, Grid]]) -> None:
    sweep_dir = ROOT / name
    for i, (p, grid) in enumerate(runs):
        save_run(sweep_dir / f"run_{i:03d}", _spectrum(p, grid), p.model_dump())
    meta = {
        "swept": swept,
        "source": "analytical all-pass model (examples/make_demo_sweep.py)",
        "n_runs": len(runs),
    }
    (sweep_dir / "sweep.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{sweep_dir}: {len(runs)} прогонов")


def main() -> None:
    fine: Grid = (1530.0, 1570.0, 0.002)
    coarse: Grid = (1530.0, 1570.0, 0.01)

    # R = 20 мкм -> FSR ~ 9 нм. a ~ 0.9971, критическая связь около t1 = 0.997.
    t1_runs = [
        (RingModelParams(ring_radius=20.0, t1=t1, **BASE), fine)
        for t1 in (0.990, 0.994, 0.997, 0.9985)
    ]
    t1_runs.append((RingModelParams(ring_radius=20.0, t1=0.9992, **BASE), coarse))
    _write_sweep("demo_t1", "t1", t1_runs)

    # R = 200 мкм -> FSR ~ 0.9 нм: меньше окна estimate_q и меньше 2*min_sep_nm.
    narrow: Grid = (1545.0, 1555.0, 0.002)
    small_fsr_runs = [
        (RingModelParams(ring_radius=200.0, t1=t1, **BASE), narrow) for t1 in (0.96, 0.98)
    ]
    _write_sweep("demo_small_fsr", "t1", small_fsr_runs)


if __name__ == "__main__":
    main()
