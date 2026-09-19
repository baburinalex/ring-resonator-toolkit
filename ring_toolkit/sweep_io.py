"""Контракт данных свипа: запись на машине с Lumerical, чтение где угодно."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .simulation import RingSpectrum

SPECTRUM_FILE = "spectrum.npz"
PARAMS_FILE = "params.json"
_ARRAYS = ("lam_nm", "t_through", "t_in")


def save_run(run_dir: str | Path, spectrum: RingSpectrum, params: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / SPECTRUM_FILE, **{k: getattr(spectrum, k) for k in _ARRAYS}
    )
    (run_dir / PARAMS_FILE).write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def load_run(run_dir: str | Path) -> tuple[RingSpectrum, dict]:
    run_dir = Path(run_dir)
    with np.load(run_dir / SPECTRUM_FILE) as z:
        missing = set(_ARRAYS) - set(z.files)
        if missing:
            raise ValueError(f"{run_dir}: в spectrum.npz нет массивов {sorted(missing)}")
        spectrum = RingSpectrum(**{k: z[k] for k in _ARRAYS})
    params = json.loads((run_dir / PARAMS_FILE).read_text(encoding="utf-8"))
    return spectrum, params


def list_runs(sweep_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(sweep_dir).glob("run_*") if p.is_dir())
