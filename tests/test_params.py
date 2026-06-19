"""Тесты валидации параметров и генерации lsf-геометрии (без lumapi)."""

import pytest
from pydantic import ValidationError

from ring_toolkit.geometry import draw_ring_geometry
from ring_toolkit.params import Platform, RingParams, SimParams


def test_ring_params_defaults_and_doubles():
    p = RingParams()
    d = p.doubles()
    assert d["wg_height"] == 0.22
    assert d["ring_radius"] == 5.0
    assert d["add_drop"] == 0.0
    assert set(p.strings()) == {"material_core", "material_clad"}


def test_negative_radius_rejected():
    with pytest.raises(ValidationError):
        RingParams(ring_radius=-1.0)


def test_add_drop_range():
    with pytest.raises(ValidationError):
        RingParams(add_drop=2)
    assert RingParams(add_drop=1).add_drop == 1


def test_sim_wavelength_order():
    with pytest.raises(ValueError):
        SimParams(lam_start=1600.0, lam_stop=1500.0)


def test_platform_override():
    p = RingParams(platform=Platform(wg_height=0.3, material_core="Si (Silicon) - Palik"))
    assert p.doubles()["wg_height"] == 0.3
    assert p.strings()["material_core"] == "Si (Silicon) - Palik"


def test_default_platform_is_soi():
    p = RingParams()
    assert p.strings()["material_core"].startswith("Si (Silicon)")
    assert p.doubles()["wg_height"] == 0.22
    assert p.wg_width == 0.5  # одномодовый strip SOI


def test_platform_preset_soi():
    assert Platform.soi().material_core.startswith("Si (Silicon)")
    assert Platform.soi().material_clad.startswith("SiO2")


def test_ring_geometry_contains_expected_commands():
    lsf = draw_ring_geometry()
    assert "ring coupler" in lsf
    assert "addring" in lsf
    assert "bus_through" in lsf
    assert "add_drop == 1" in lsf  # ветка add-drop присутствует



