import json
import math

import numpy as np
import pytest

from ring_toolkit.analysis import Resonance, SpectrumAnalysis, analyze_spectrum
from ring_toolkit.analyze import check_analysis, main
from ring_toolkit.simulation import RingSpectrum
from ring_toolkit.sweep_io import save_run

LAM = np.arange(1540.0, 1560.0, 0.01)  # шаг 0.01 нм


def _codes(result: dict) -> set[str]:
    return {a["code"] for a in result["anomalies"]}


def test_clean_case_has_no_anomalies():
    # Q=1e4 -> FWHM 0.155 нм -> ~15 точек на FWHM; FSR 8 нм больше окна и 2*min_sep
    analysis = SpectrumAnalysis(
        resonances=[Resonance(lambda0_nm=1546.0, q=1e4, depth=0.8),
                    Resonance(lambda0_nm=1554.0, q=1e4, depth=0.8)],
        fsr_nm=[8.0],
    )
    # R = 20 мкм: по геометрии в 20 нм ожидается >= ~1.6 резонанса, найдено 2 — норма
    result = check_analysis(
        analysis, LAM, win_nm=5.0, min_sep_nm=1.0, round_trip_um=2 * math.pi * 20.0
    )
    assert result["anomalies"] == []
    assert all(r["q_reliable"] for r in result["resonances"])


def test_single_resonance_flags_fsr_undetermined():
    analysis = SpectrumAnalysis(resonances=[Resonance(lambda0_nm=1550.0, q=1e4, depth=0.8)])
    assert "FSR_UNDETERMINED" in _codes(check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0))


def test_too_few_resonances_for_geometry_is_flagged():
    # R = 200 мкм: в 20 нм ожидается >= ~15 резонансов, найден один
    analysis = SpectrumAnalysis(resonances=[Resonance(lambda0_nm=1550.0, q=1e4, depth=0.8)])
    result = check_analysis(
        analysis, LAM, win_nm=5.0, min_sep_nm=1.0, round_trip_um=2 * math.pi * 200.0
    )
    assert "RESONANCE_COUNT_LOW" in _codes(result)


def test_undersampled_high_q_is_flagged_not_reported_as_reliable():
    # Q=1e6 -> FWHM 0.00155 нм при шаге 0.01 нм: меньше одной точки на линию
    analysis = SpectrumAnalysis(resonances=[Resonance(lambda0_nm=1550.0, q=1e6, depth=0.5)])
    result = check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0)
    assert "UNDERSAMPLED_FWHM" in _codes(result)
    assert result["resonances"][0]["q_reliable"] is False
    assert result["mean_q_loaded_reliable"] is None


def test_failed_q_is_counted_and_json_safe():
    analysis = SpectrumAnalysis(
        resonances=[Resonance(lambda0_nm=1550.0, q=float("nan"), depth=0.3)]
    )
    result = check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0)
    assert result["n_q_failed"] == 1
    assert "Q_FAILED" in _codes(result)
    assert "NaN" not in json.dumps(result)


def test_small_fsr_flags_window_and_min_sep():
    analysis = SpectrumAnalysis(
        resonances=[Resonance(lambda0_nm=1550.0 + 0.6 * i, q=1e4, depth=0.8) for i in range(3)],
        fsr_nm=[0.6, 0.6],
    )
    codes = _codes(check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0))
    assert {"WINDOW_GE_FSR", "MIN_SEP_NEAR_FSR"} <= codes


def test_nonuniform_fsr_is_flagged():
    analysis = SpectrumAnalysis(
        resonances=[Resonance(lambda0_nm=x, q=1e4, depth=0.8) for x in (1542.0, 1550.0, 1566.0)],
        fsr_nm=[8.0, 16.0],
    )
    assert "FSR_NONUNIFORM" in _codes(check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0))


def _lorentzian_spectrum(shift_nm: float) -> RingSpectrum:
    t = np.ones_like(LAM)
    for lam0 in np.arange(1542.0, 1560.0, 4.0) + shift_nm:
        t -= 0.8 / (1.0 + ((LAM - lam0) / 0.05) ** 2)
    return RingSpectrum(lam_nm=LAM, t_through=t, t_in=np.ones_like(LAM))


SHAPE_CODES = {"RESONANCE_SPLITTING", "BASELINE_TILT", "BASELINE_RIPPLE"}
CENTERS = np.arange(1542.0, 1560.0, 4.0)  # FSR 4 нм, FWHM 0.1 нм (10 точек сетки)


def _dips(split_nm: float = 0.0) -> np.ndarray:
    """Лоренцевы провалы; split_nm > 0 — каждый расщеплён в дублет."""
    t = np.ones_like(LAM)
    for lam0 in CENTERS:
        for c in ((lam0 - split_nm / 2, lam0 + split_nm / 2) if split_nm else (lam0,)):
            t -= 0.8 / (1.0 + ((LAM - c) / 0.05) ** 2) / (2 if split_nm else 1)
    return t


def _noisy(t: np.ndarray, seed: int = 0) -> np.ndarray:
    return t + np.random.default_rng(seed).normal(0.0, 0.002, t.shape)


def _shape_codes(t: np.ndarray, lam: np.ndarray = LAM) -> set[str]:
    analysis = analyze_spectrum(lam, t)
    result = check_analysis(analysis, lam, win_nm=5.0, min_sep_nm=1.0, t=t)
    return _codes(result) & SHAPE_CODES


@pytest.mark.parametrize("seed", range(5))
def test_clean_spectrum_has_no_shape_anomalies(seed):
    assert _shape_codes(_noisy(_dips(), seed)) == set()


@pytest.mark.parametrize("seed", range(5))
def test_doublet_is_flagged(seed):
    # компоненты на расстоянии 0.25 нм = 2.5 FWHM; find_resonances сливает их в один провал
    assert _shape_codes(_noisy(_dips(split_nm=0.25), seed)) == {"RESONANCE_SPLITTING"}


@pytest.mark.parametrize("seed", range(5))
def test_baseline_tilt_is_flagged(seed):
    tilt = 1.0 - 0.2 * (LAM - LAM[0]) / np.ptp(LAM)
    assert _shape_codes(_noisy(0.9 * tilt * _dips(), seed)) == {"BASELINE_TILT"}


@pytest.mark.parametrize("seed", range(5))
def test_fabry_perot_ripple_is_flagged(seed):
    ripple = 1.0 + 0.06 * np.cos(2 * np.pi * LAM / 1.2)  # период 1.2 нм = 0.3 FSR
    assert _shape_codes(_noisy(0.9 * ripple * _dips(), seed)) == {"BASELINE_RIPPLE"}


def test_shape_checks_survive_repeated_samples():
    lam = np.repeat(LAM, 2)
    t = _noisy(np.repeat(_dips(split_nm=0.25), 2))
    assert _shape_codes(t, lam) == {"RESONANCE_SPLITTING"}


def test_shape_checks_need_spectrum():
    # без t (старый вызов) проверки формы не выполняются
    t = _noisy(_dips(split_nm=0.25))
    result = check_analysis(analyze_spectrum(LAM, t), LAM, win_nm=5.0, min_sep_nm=1.0)
    assert _codes(result) & SHAPE_CODES == set()


def test_cli_reports_shape_anomalies(tmp_path):
    ripple = 1.0 + 0.06 * np.cos(2 * np.pi * LAM / 1.2)
    t = _noisy(0.9 * ripple * _dips())
    spec = RingSpectrum(lam_nm=LAM, t_through=t, t_in=np.ones_like(LAM))
    save_run(tmp_path / "run_000", spec, {"ring_radius": 20.0})
    assert main([str(tmp_path)]) == 0
    report = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert "BASELINE_RIPPLE" in {a["code"] for a in report["runs"][0]["anomalies"]}


def test_cli_writes_analysis_json(tmp_path, capsys):
    for i, shift in enumerate((0.0, 0.3)):
        save_run(tmp_path / f"run_{i:03d}", _lorentzian_spectrum(shift), {"gap_um": 0.4 + 0.1 * i})

    assert main([str(tmp_path)]) == 0

    report = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert report["n_runs"] == 2
    assert [r["run"] for r in report["runs"]] == ["run_000", "run_001"]
    assert report["runs"][1]["params"] == {"gap_um": 0.5}
    assert isinstance(report["runs"][0]["anomalies"], list)
    assert "Записано" in capsys.readouterr().out


def test_cli_empty_dir_returns_2(tmp_path):
    assert main([str(tmp_path)]) == 2
