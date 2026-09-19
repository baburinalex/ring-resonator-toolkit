import json

import numpy as np

from ring_toolkit.analysis import Resonance, SpectrumAnalysis
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
    result = check_analysis(analysis, LAM, win_nm=5.0, min_sep_nm=1.0)
    assert result["anomalies"] == []
    assert all(r["q_reliable"] for r in result["resonances"])


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
