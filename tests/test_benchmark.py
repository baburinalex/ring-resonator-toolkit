import json

import numpy as np
import pytest

from ring_toolkit.analytical_model import (
    RingModelParams,
    all_pass_transmission,
    figures_of_merit,
)
from ring_toolkit.benchmark import (
    ANOMALY_CODES,
    REGIMES,
    TRAPS,
    classify_regime,
    generate_benchmark,
    main,
    make_case,
    through_field,
    write_case,
)
from ring_toolkit.sweep_io import list_runs, load_run

SEEDS = (0, 1, 7)
TRUTH_KEYS = {
    "case_id", "trap", "regime", "q_i", "q_c", "kappa2_1550", "fsr_nm",
    "expected_anomalies", "notes",
}


def _model(case) -> RingModelParams:
    return RingModelParams(**case.truth["generator"]["model"])


def _dips(lam, t, min_depth=0.1, min_sep_nm=0.0):
    """Индексы локальных минимумов глубиной >= min_depth от верхней огибающей."""
    from scipy.signal import find_peaks

    step = np.median(np.diff(np.unique(lam)))
    dist = max(1, int(min_sep_nm / step)) if min_sep_nm else 1
    idx, _ = find_peaks(-t, prominence=min_depth, distance=dist)
    return idx


def _fwhm_nm(lam, t, idx):
    """Ширина провала на полувысоте между минимумом и единицей (линейная интерполяция)."""
    half = 0.5 * (1.0 + t[idx])

    def _edge(step):
        i = idx
        while 0 < i < len(t) - 1 and t[i] < half:
            i += step
        j = i - step  # последний отсчёт ниже полувысоты
        return lam[j] + (half - t[j]) * (lam[i] - lam[j]) / (t[i] - t[j])

    return _edge(1) - _edge(-1)


# ----------------------------------------------------------------------
# Общие свойства для всех ловушек
# ----------------------------------------------------------------------
@pytest.mark.parametrize("trap", TRAPS)
@pytest.mark.parametrize("seed", SEEDS)
def test_physical_and_well_formed(trap, seed):
    case = make_case(trap, seed)
    s = case.spectrum
    assert s.lam_nm.shape == s.t_through.shape == s.t_in.shape
    assert np.all(np.isfinite(s.t_through))
    assert np.all((s.t_through >= 0.0) & (s.t_through <= 1.0))
    assert np.all(np.diff(s.lam_nm) >= 0)

    truth = case.truth
    assert TRUTH_KEYS <= truth.keys()
    assert truth["trap"] == trap
    assert truth["regime"] in REGIMES
    assert set(truth["expected_anomalies"]) <= ANOMALY_CODES.keys()
    assert truth["q_i"] > 0 and truth["q_c"] > 0 and 0 < truth["kappa2_1550"] < 1
    assert truth["notes"]


@pytest.mark.parametrize("trap", TRAPS)
@pytest.mark.parametrize("seed", SEEDS)
def test_truth_reproduces_from_model(trap, seed):
    """Истина пересчитывается из сохранённой модели и согласована с режимом."""
    case = make_case(trap, seed)
    truth, p = case.truth, _model(case)
    fom = figures_of_merit(p)
    assert truth["q_i"] == pytest.approx(fom.q_intrinsic)
    assert truth["q_c"] == pytest.approx(fom.q_coupling)
    assert truth["fsr_nm"] == pytest.approx(fom.fsr_nm)
    assert truth["kappa2_1550"] == pytest.approx(1 - p.t1**2)
    rho = truth["kappa2_1550"] / (1 - p.amplitude_a**2)
    assert classify_regime(rho) == truth["regime"]
    # пересвязь <=> Q_c < Q_i
    if truth["regime"] == "overcoupled":
        assert truth["q_c"] < truth["q_i"]
    if truth["regime"] == "undercoupled":
        assert truth["q_c"] > truth["q_i"]


@pytest.mark.parametrize("trap", TRAPS)
def test_deterministic_by_seed(trap):
    a, b, c = make_case(trap, 3), make_case(trap, 3), make_case(trap, 4)
    np.testing.assert_array_equal(a.spectrum.lam_nm, b.spectrum.lam_nm)
    np.testing.assert_array_equal(a.spectrum.t_through, b.spectrum.t_through)
    assert json.dumps(a.truth) == json.dumps(b.truth)
    assert a.params == b.params
    assert a.truth["q_i"] != c.truth["q_i"]


def test_params_do_not_leak_truth():
    """params.json содержит только то, что знает экспериментатор."""
    for trap in TRAPS:
        case = make_case(trap, 0)
        assert set(case.params) == {
            "ring_radius", "coupling_length", "lam0_nm", "kappa2_design", "port",
        }
        # проектная оценка ближе к истинной kappa^2, чем к потерям -> ветка выбирается
        p = _model(case)
        k2, loss = case.truth["kappa2_1550"], 1 - p.amplitude_a**2
        design = case.params["kappa2_design"]
        assert abs(np.log(design / k2)) <= np.log(1.25) + 1e-12
        if case.truth["regime"] != "critical":
            assert abs(np.log(design / k2)) < abs(np.log(design / loss))


def test_through_field_matches_analytical_model():
    p = RingModelParams(ring_radius=20.0, t1=0.99, loss_db_cm=3.0, n_eff0=1.8, n_g0=2.1)
    lam = np.linspace(1540, 1560, 5001)
    np.testing.assert_allclose(
        np.abs(through_field(lam, p)) ** 2, all_pass_transmission(lam, p), atol=1e-12
    )


# ----------------------------------------------------------------------
# Сигнатура каждой ловушки видна в данных
# ----------------------------------------------------------------------
@pytest.mark.parametrize("trap", ["overcoupled", "undercoupled", "critical"])
@pytest.mark.parametrize("seed", SEEDS)
def test_regime_cases_linewidth_and_depth(trap, seed):
    """Чистые случаи: ширина линии даёт Q_L, глубина — T_min модели."""
    case = make_case(trap, seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    fom = figures_of_merit(_model(case))
    idx = _dips(lam, t, min_depth=0.3)
    assert len(idx) >= 3
    i = idx[np.argmin(np.abs(lam[idx] - 1550.0))]
    assert 1550.0 / _fwhm_nm(lam, t, i) == pytest.approx(fom.q_loaded, rel=0.1)
    assert t[i] == pytest.approx(fom.t_min, abs=0.01)
    assert np.diff(lam[idx]).mean() == pytest.approx(fom.fsr_nm, rel=0.02)


@pytest.mark.parametrize("seed", SEEDS)
def test_undersampled_step_near_fwhm(seed):
    case = make_case("undersampled", seed)
    step = np.median(np.diff(case.spectrum.lam_nm))
    fwhm = 1550.0 / case.truth["q_loaded"]
    assert 0.75 <= step / fwhm <= 1.25


@pytest.mark.parametrize("seed", SEEDS)
def test_duplicate_samples_repeat_wavelengths(seed):
    case = make_case("duplicate_samples", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    k = case.truth["generator"]["trap_params"]["repeats"]
    assert np.median(np.diff(lam)) == 0.0
    assert len(lam) == k * len(np.unique(lam))
    # повторы — разные отсчёты шума, а не копии
    assert not np.array_equal(t[0::k], t[1::k])


@pytest.mark.parametrize("seed", SEEDS)
def test_doublet_is_split(seed):
    case = make_case("doublet", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    fsr = case.truth["fsr_nm"]
    fwhm = 1550.0 / case.truth["q_loaded"]
    # около каждого резонанса — два минимума на расстоянии ~ split * FWHM
    mins = lam[_dips(lam, t, min_depth=0.05)]
    center = mins[np.argmin(np.abs(mins - 1550.0))]
    pair = np.sort(mins[np.abs(mins - center) < min(3 * fwhm, 0.25 * fsr)])
    assert len(pair) == 2
    split = case.truth["generator"]["trap_params"]["split_over_fwhm"]
    assert (pair[1] - pair[0]) / fwhm == pytest.approx(split, rel=0.3)


@pytest.mark.parametrize("seed", SEEDS)
def test_baseline_tilt_and_ripple(seed):
    case = make_case("baseline", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    tp = case.truth["generator"]["trap_params"]
    # верхняя огибающая в левой и правой четверти отличается по знаку наклона
    q = len(lam) // 4
    left, right = t[:q].max(), t[-q:].max()
    assert np.sign(right - left) == np.sign(tp["tilt"])
    assert abs(right - left) > 0.05
    assert t.max() < 1.0  # базовая линия ниже единицы: спектр не нормирован
    # рябь: между резонансами сигнал гуляет заметнее шума
    m = np.abs(lam - (1550.0 + case.truth["fsr_nm"] / 2)) < tp["ripple_period_nm"]
    assert np.ptp(t[m]) > 5 * 0.002


@pytest.mark.parametrize("seed", SEEDS)
def test_kappa_dispersion_changes_depth(seed):
    case = make_case("kappa_dispersion", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    k0, k1 = case.truth["generator"]["trap_params"]["kappa2_edges"]
    assert abs(k1 - k0) / case.truth["kappa2_1550"] > 0.5
    idx = _dips(lam, t, min_depth=0.1)
    depths = t[idx]
    assert np.ptp(depths) > 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_two_families_two_fsr(seed):
    case = make_case("two_families", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    fsr1 = case.truth["fsr_nm"]
    fsr2 = case.truth["generator"]["trap_params"]["family2_fsr_nm"]
    assert abs(fsr2 - fsr1) / fsr1 > 0.05
    span = np.ptp(lam)
    n_dips = len(_dips(lam, t, min_depth=0.05))
    # резонансов больше, чем даёт одно семейство
    assert n_dips > span / fsr1 + 1


@pytest.mark.parametrize("seed", SEEDS)
def test_small_fsr_below_1nm(seed):
    case = make_case("small_fsr", seed)
    lam, t = case.spectrum.lam_nm, case.spectrum.t_through
    assert case.truth["fsr_nm"] < 1.0
    idx = _dips(lam, t, min_depth=0.1)
    assert len(idx) >= 8
    assert np.median(np.diff(lam[idx])) == pytest.approx(case.truth["fsr_nm"], rel=0.02)


# ----------------------------------------------------------------------
# Запись на диск
# ----------------------------------------------------------------------
def test_write_case_sweep_io_format(tmp_path):
    case = make_case("doublet", 0)
    case_dir = write_case(case, tmp_path)
    assert [p.name for p in list_runs(case_dir)] == ["run_000"]
    spectrum, params = load_run(case_dir / "run_000")
    np.testing.assert_array_equal(spectrum.t_through, case.spectrum.t_through)
    assert params == case.params
    assert json.loads((case_dir / "truth.json").read_text(encoding="utf-8")) == case.truth
    meta = json.loads((case_dir / "sweep.json").read_text(encoding="utf-8"))
    assert "doublet" not in json.dumps(meta)


def test_generate_benchmark_neutral_names_and_manifest(tmp_path):
    paths = generate_benchmark(tmp_path, seed=5, n_per_trap=2)
    assert len(paths) == 2 * len(TRAPS)
    assert all(p.name.startswith("case_") for p in paths)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for entry, path in zip(manifest["cases"], paths, strict=True):
        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        assert truth["case_id"] == entry["case_id"] == path.name
        assert truth["trap"] == entry["trap"]
    # повторная генерация — те же файлы
    again = tmp_path / "again"
    generate_benchmark(again, seed=5, n_per_trap=2)
    for path in paths:
        assert (path / "truth.json").read_bytes() == (again / path.name / "truth.json").read_bytes()


def test_cli(tmp_path, capsys):
    assert main([str(tmp_path), "--seed", "2", "--traps", "critical", "small_fsr"]) == 0
    assert "Записано случаев: 2" in capsys.readouterr().out
    assert len(list(tmp_path.glob("case_*"))) == 2


def test_unknown_trap():
    with pytest.raises(ValueError, match="неизвестная ловушка"):
        make_case("nope", 0)
