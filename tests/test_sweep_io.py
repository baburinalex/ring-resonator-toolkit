import numpy as np
import pytest

from ring_toolkit.simulation import RingSpectrum
from ring_toolkit.sweep_io import list_runs, load_run, save_run


def _spectrum() -> RingSpectrum:
    lam = np.linspace(1549.0, 1551.0, 201)
    return RingSpectrum(lam_nm=lam, t_through=np.full_like(lam, 0.9), t_in=np.ones_like(lam))


def test_roundtrip(tmp_path):
    spec = _spectrum()
    save_run(tmp_path / "run_000", spec, {"gap_um": 0.5, "заметка": "тест"})
    out, params = load_run(tmp_path / "run_000")
    np.testing.assert_array_equal(out.lam_nm, spec.lam_nm)
    np.testing.assert_array_equal(out.t_norm, spec.t_norm)
    assert params == {"gap_um": 0.5, "заметка": "тест"}


def test_missing_array_raises(tmp_path):
    run = tmp_path / "run_000"
    run.mkdir()
    np.savez_compressed(run / "spectrum.npz", lam_nm=np.arange(3.0))
    (run / "params.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="t_in"):
        load_run(run)


def test_list_runs_sorted(tmp_path):
    for name in ("run_002", "run_000", "run_001", "notes"):
        (tmp_path / name).mkdir()
    assert [p.name for p in list_runs(tmp_path)] == ["run_000", "run_001", "run_002"]
