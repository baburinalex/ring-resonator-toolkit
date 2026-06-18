"""Тесты аналитического оценщика геометрии 1x2 MMI (без lumapi)."""

import numpy as np

from lumfdtd.design import (
    beat_length_um,
    mmi_1x2_symmetric,
    suggest_length_sweep_um,
)


def test_beat_length_scales_with_width_squared():
    # При фиксированных lambda и n_eff: L_pi ~ W^2
    l1 = beat_length_um(3.0)
    l2 = beat_length_um(6.0)
    assert abs(l2 / l1 - 4.0) < 1e-9


def test_soi_1x2_estimate_reasonable():
    est = mmi_1x2_symmetric(3.0, wavelength_um=1.55, n_eff=2.85)
    # для W=3 мкм на SOI ожидаем L_pi ~ 22..27 мкм, L_mmi ~ 7..11 мкм
    assert 20.0 < est.L_pi_um < 28.0
    assert 7.0 < est.L_mmi_um < 11.0
    assert est.out_sep_um == 1.5          # центры выходов на ± W/4
    assert est.regime == "symmetric"


def test_length_relation_3lpi_over_8():
    est = mmi_1x2_symmetric(3.0)
    assert abs(est.L_mmi_um - 3.0 * est.L_pi_um / 8.0) < 1e-9


def test_suggest_sweep_brackets_estimate():
    est = mmi_1x2_symmetric(3.0)
    sweep = suggest_length_sweep_um(est.L_mmi_um, span_frac=0.35, n=15)
    assert len(sweep) == 15
    assert sweep[0] < est.L_mmi_um < sweep[-1]
    assert np.all(np.diff(sweep) > 0)
