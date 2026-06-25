"""Тесты аналитической модели кольца (чистый numpy, без lumapi)."""

import numpy as np
import pytest
from ring_toolkit.analytical_model import (
    RingModelParams,
    add_drop_transmission,
    all_pass_transmission,
    figures_of_merit,
)

from ring_toolkit.analysis import analyze_spectrum


def _soi_params(**kw) -> RingModelParams:
    base = dict(ring_radius=5.0, n_eff0=2.4, n_g0=4.2, lam0_nm=1550.0, loss_db_cm=3.0, t1=0.9)
    base.update(kw)
    return RingModelParams(**base)


def test_transmission_bounded():
    p = _soi_params(loss_db_cm=50.0)
    lam = np.linspace(1500.0, 1600.0, 20000)
    t = all_pass_transmission(lam, p)
    assert np.all(t >= -1e-9) and np.all(t <= 1.0 + 1e-9)


def test_critical_coupling_extinguishes():
    """t1 == a -> аналитический T_min == 0 (критическая связь)."""
    base = _soi_params(loss_db_cm=20.0)
    p = _soi_params(loss_db_cm=20.0, t1=base.amplitude_a)
    assert figures_of_merit(p).t_min < 1e-9


def test_undercoupled_vs_overcoupled_tmin():
    """Перекос связи в обе стороны от критической поднимает T_min."""
    a = _soi_params().amplitude_a
    t_under = figures_of_merit(_soi_params(t1=min(a * 1.3, 0.999))).t_min
    t_crit = figures_of_merit(_soi_params(t1=a)).t_min
    t_over = figures_of_merit(_soi_params(t1=a * 0.7)).t_min
    assert t_crit < t_under and t_crit < t_over


def test_fsr_self_consistent_with_analysis():
    """Аналитический спектр -> analyze_spectrum -> измеренный FSR == lam^2/(n_g L)."""
    p = _soi_params(loss_db_cm=50.0)
    lam = np.linspace(1500.0, 1600.0, 60000)
    t = all_pass_transmission(lam, p)
    res = analyze_spectrum(lam, t)
    analytic_fsr = figures_of_merit(p).fsr_nm
    assert res.mean_fsr_nm == pytest.approx(analytic_fsr, rel=0.02)


def test_q_recovered_from_spectrum():
    """Q из спектра (analyze_spectrum) ~ аналитической Q_loaded для глубокого резонанса."""
    base = _soi_params(loss_db_cm=60.0)
    p = _soi_params(loss_db_cm=60.0, t1=round(base.amplitude_a, 4))  # около критической связи
    lam = np.linspace(1540.0, 1560.0, 8000)
    t = all_pass_transmission(lam, p)
    res = analyze_spectrum(lam, t)
    assert res.resonances
    q_meas = res.resonances[0].q
    assert q_meas == pytest.approx(figures_of_merit(p).q_loaded, rel=0.15)


def test_add_drop_drop_peaks_at_resonance():
    """В резонансе drop максимален, through минимален; вне — наоборот."""
    p = _soi_params(t1=0.98, t2=0.98)
    lam = np.linspace(1500.0, 1600.0, 60000)
    t_pass, t_drop = add_drop_transmission(lam, p)
    i = int(np.argmax(t_drop))
    assert t_pass[i] < 0.2
    assert t_drop.max() > 0.5


def test_add_drop_requires_t2():
    with pytest.raises(ValueError):
        add_drop_transmission(np.array([1550.0]), _soi_params())  # t2 is None


def test_q_decomposition():
    """1/Q_l = 1/Q_i + 1/Q_c и Q_l <= обоих."""
    fom = figures_of_merit(_soi_params(t1=0.95, loss_db_cm=2.0))
    assert fom.q_loaded <= fom.q_intrinsic + 1
    assert fom.q_loaded <= fom.q_coupling + 1
    inv = 1.0 / fom.q_intrinsic + 1.0 / fom.q_coupling
    assert (1.0 / fom.q_loaded) == pytest.approx(inv, rel=1e-6)
