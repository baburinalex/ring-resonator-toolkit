"""
ring_toolkit — проектирование, моделирование и анализ фотонных кольцевых
резонаторов на платформе SOI (кремний-на-изоляторе, 220 нм).

Два слоя:
    layout (GDS): параметрическая раскладка на gdsfactory —
        ring_toolkit.components (RingResonatorParams, ring_resonator_all_pass).
    моделирование: FDTD-цепочка через Ansys Lumerical (lumapi) и анализ
        спектра. Платформа задаётся через Platform.soi().

Публичный API верхнего уровня (лёгкий импорт, без gdsfactory):
    params:   RingParams, Platform, SimParams
    geometry: draw_ring_geometry
    analysis: analyze_spectrum, find_resonances, estimate_q,
              Resonance, SpectrumAnalysis

Запуск FDTD — из ring_toolkit.simulation (run_ring, RingSpectrum):
вынесен из пакетного __init__, чтобы импорт не тянул lumapi.
Раскладка слоёв — из ring_toolkit.components (требует gdsfactory).
"""

from __future__ import annotations

from .analysis import (
    Resonance,
    SpectrumAnalysis,
    analyze_spectrum,
    estimate_q,
    find_resonances,
)
from .geometry import draw_ring_geometry
from .params import Platform, RingParams, SimParams

__version__ = "0.1.0"

__all__ = [
    "Platform",
    "RingParams",
    "Resonance",
    "SimParams",
    "SpectrumAnalysis",
    "analyze_spectrum",
    "draw_ring_geometry",
    "estimate_q",
    "find_resonances",
]
