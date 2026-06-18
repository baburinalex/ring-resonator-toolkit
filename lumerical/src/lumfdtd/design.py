"""
Аналитические оценки геометрии 1x2 MMI-делителя (теория самовоспроизведения).

Геометрия в этом проекте имеет вход по центру (y = 0) -> это режим
СИММЕТРИЧНОЙ интерференции. Для него:

    длина биений   L_pi = 4 * n_eff * W_e^2 / (3 * lambda)
    длина 1xN MMI  L_mmi = 3 * L_pi / (4 * N)          (для N=2: 3*L_pi/8)
    положения выходов: y = ± W_e / (2N)                (для N=2: ± W_e/4)

где W_e ~ W_mmi (+ небольшая поправка на проникновение поля, для
сильного контраста SOI ~ десятки нм на сторону), n_eff — эффективный
индекс фундаментальной моды широкой области MMI (для 220-нм SOI TE
близок к индексу плёнки ~2.85).

Оценка — стартовая точка; точную длину доводят свипом FDTD
(run_mmi_sweep), потому что W_e и n_eff зависят от платформы и
дисперсии. Модуль не зависит от lumapi и покрыт тестами.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MMIDesignEstimate:
    L_pi_um: float          # длина биений
    L_mmi_um: float         # оценка длины области MMI
    out_sep_um: float       # расстояние центр-в-центр между выходами
    regime: str

    def __str__(self) -> str:
        return (
            f"MMI 1x2 ({self.regime}): L_pi = {self.L_pi_um:.2f} мкм, "
            f"L_mmi ~ {self.L_mmi_um:.2f} мкм, "
            f"расстояние между выходами ~ {self.out_sep_um:.2f} мкм"
        )


def beat_length_um(
    w_eff_um: float, wavelength_um: float = 1.55, n_eff: float = 2.85
) -> float:
    """Длина биений L_pi = 4 n_eff W_e^2 / (3 lambda), мкм."""
    return 4.0 * n_eff * w_eff_um**2 / (3.0 * wavelength_um)


def mmi_1x2_symmetric(
    w_mmi_um: float,
    wavelength_um: float = 1.55,
    n_eff: float = 2.85,
    we_pad_um: float = 0.1,
) -> MMIDesignEstimate:
    """Оценка геометрии 1x2 MMI для симметричной интерференции (вход по центру)."""
    w_eff = w_mmi_um + we_pad_um
    l_pi = beat_length_um(w_eff, wavelength_um, n_eff)
    l_mmi = 3.0 * l_pi / 8.0          # N = 2
    out_sep = w_mmi_um / 2.0          # центры выходов на ± W/4
    return MMIDesignEstimate(l_pi, l_mmi, out_sep, "symmetric")


def suggest_length_sweep_um(
    l_mmi_um: float, span_frac: float = 0.35, n: int = 15
) -> np.ndarray:
    """Диапазон длин для FDTD-свипа вокруг аналитической оценки."""
    lo = l_mmi_um * (1.0 - span_frac)
    hi = l_mmi_um * (1.0 + span_frac)
    return np.linspace(lo, hi, n)
