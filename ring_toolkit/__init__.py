"""
lumfdtd — моделирование фотонных компонентов на SOI (кремний-на-изоляторе)
в Ansys Lumerical FDTD (1x2 MMI-делитель и кольцевой резонатор) через lumapi.
Платформа настраивается: Platform.soi() / Platform.sin().

Публичный API:
    params:   RingParams, MMIParams, Platform, SimParams
    geometry: draw_ring_geometry, draw_mmi_geometry
    sim:      run_ring, run_mmi_sweep, RingSpectrum
    analysis: analyze_spectrum, find_resonances, estimate_q,
              Resonance, SpectrumAnalysis
"""

from __future__ import annotations

from .analysis import (
    Resonance,
    SpectrumAnalysis,
    analyze_spectrum,
    estimate_q,
    find_resonances,
)
from .design import (
    MMIDesignEstimate,
    beat_length_um,
    mmi_1x2_symmetric,
    suggest_length_sweep_um,
)
from .geometry import draw_mmi_geometry, draw_ring_geometry
from .params import MMIParams, Platform, RingParams, SimParams

__version__ = "0.1.0"

__all__ = [
    "MMIDesignEstimate",
    "MMIParams",
    "Platform",
    "RingParams",
    "Resonance",
    "SimParams",
    "SpectrumAnalysis",
    "analyze_spectrum",
    "beat_length_um",
    "draw_mmi_geometry",
    "draw_ring_geometry",
    "estimate_q",
    "find_resonances",
    "mmi_1x2_symmetric",
    "suggest_length_sweep_um",
]
