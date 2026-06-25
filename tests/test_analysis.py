"""Тесты анализа спектра (без lumapi): синтетика + регрессия на узкие резонансы с рябью."""

import numpy as np

from ring_toolkit.analysis import (
    analyze_spectrum,
    estimate_q,
    find_resonances,
    group_index_from_fsr,
)


def lorentzian_dip(lam_nm, lam0, fwhm, depth=0.9):
    """T(lambda) = 1 - depth / (1 + ((lam-lam0)/(fwhm/2))^2)."""
    hwhm = fwhm / 2.0
    return 1.0 - depth / (1.0 + ((lam_nm - lam0) / hwhm) ** 2)


def test_single_resonance_q_recovered():
    lam = np.arange(1500.0, 1600.0, 0.005)
    lam0, fwhm = 1550.0, 0.5  # ожидаемый Q = 1550 / 0.5 = 3100
    t = lorentzian_dip(lam, lam0, fwhm)
    dips = find_resonances(lam, t)
    assert len(dips) == 1
    l0, q, depth = estimate_q(lam, t, dips[0])
    assert abs(l0 - lam0) < 0.05
    assert abs(q - 3100.0) / 3100.0 < 0.1
    assert depth > 0.5


def test_two_resonances_fsr():
    lam = np.arange(1500.0, 1600.0, 0.005)
    t = lorentzian_dip(lam, 1540.0, 0.4) * lorentzian_dip(lam, 1560.0, 0.4)
    res = analyze_spectrum(lam, t)
    assert len(res.resonances) == 2
    assert len(res.fsr_nm) == 1
    assert abs(res.fsr_nm[0] - 20.0) < 0.2
    assert abs(res.mean_fsr_nm - 20.0) < 0.2


def test_flat_spectrum_no_resonances():
    lam = np.linspace(1500.0, 1600.0, 1000)
    t = np.ones_like(lam) * 0.95
    res = analyze_spectrum(lam, t)
    assert res.resonances == []
    assert np.isnan(res.mean_fsr_nm)


def test_unsorted_input_handled():
    lam = np.arange(1500.0, 1600.0, 0.01)
    t = lorentzian_dip(lam, 1550.0, 0.5)
    perm = np.random.permutation(len(lam))
    res = analyze_spectrum(lam[perm], t[perm])
    assert len(res.resonances) == 1
    assert abs(res.resonances[0].lambda0_nm - 1550.0) < 0.1


def test_narrow_resonance_with_ripple_not_split():
    # узкий глубокий провал + мелкая рябь поверх (реальный сценарий FDTD).
    # Регрессия: рябь НЕ должна давать ложных резонансов и дробить провал.
    lam = np.arange(1500.0, 1600.0, 0.1)
    t = lorentzian_dip(lam, 1550.0, 0.3, depth=0.7)
    t = t + 0.03 * np.sin(2 * np.pi * lam / 0.4)
    res = analyze_spectrum(lam, t, min_depth=0.2)
    assert len(res.resonances) == 1
    assert abs(res.resonances[0].lambda0_nm - 1550.0) < 0.3


def test_shallow_ripple_rejected():
    # только рябь, без настоящих провалов -> ничего не находим
    lam = np.arange(1500.0, 1600.0, 0.1)
    t = 1.0 + 0.04 * np.sin(2 * np.pi * lam / 0.4)
    res = analyze_spectrum(lam, t, min_depth=0.2)
    assert res.resonances == []


def test_group_index_from_fsr():
    # FSR=18.4 нм, R=5 мкм, lambda=1550 нм -> n_g ~ 4.1 (SOI 500x220)
    ng = group_index_from_fsr(18.4, 1550.0, 5.0)
    assert 3.8 < ng < 4.5
