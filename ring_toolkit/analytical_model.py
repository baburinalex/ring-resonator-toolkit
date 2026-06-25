"""
Аналитическая модель кольцевого резонатора (coupled-mode / transfer-matrix).

Замкнутые формулы из учебной нотации (Bogaerts et al., "Silicon microring
resonators", Laser Photonics Rev. 2012): спектр пропускания all-pass и add-drop
колец + figures of merit. Чистый numpy/pydantic, без lumapi.

Модель ПРИНИМАЕТ физические входы (n_eff, n_g, потери, коэффициенты связи t),
а не вычисляет их — n_eff/n_g берутся из мод-солвера, t(gap) — из калибровки
или FDTD. Это быстрое замкнутое отображение (n_eff, n_g, alpha, t, R, lam) ->
спектр + FOM, дополняющее (не заменяющее) FDTD из ring_toolkit.simulation.

Соглашения:
    - длины волн в НАНОМЕТРАХ, геометрия в МИКРОМЕТРАХ;
    - потери — мощностные, в дБ/см (loss_db_cm), внутри переводятся в alpha[1/мкм];
    - связь в нотации t/k: t — self-coupling (амплитудный), k = sqrt(1 - t^2);
    - дисперсия линейная: задаётся n_eff0 и n_g0 на опорной lam0_nm, наклон
      dn_eff/dlam = (n_eff0 - n_g0)/lam0 восстанавливается из определения n_g.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, model_validator

_DB_PER_CM_TO_NP_PER_UM = math.log(10.0) / 10.0 / 1.0e4  # power alpha: dB/cm -> Np/um


class RingModelParams(BaseModel):
    """Входы аналитической модели кольца.

    Для all-pass задаётся только ``t1``. Для add-drop — ``t1`` и ``t2``.
    """

    # геометрия (мкм)
    ring_radius: float = Field(..., gt=0, description="радиус кольца, мкм")
    coupling_length: float = Field(
        0.0, ge=0, description="длина прямого участка связи (racetrack), мкм; 0 = чистое кольцо"
    )

    # дисперсия (линейная около lam0)
    n_eff0: float = Field(..., gt=0, description="фазовый индекс на lam0")
    n_g0: float = Field(..., gt=0, description="групповой индекс на lam0")
    lam0_nm: float = Field(1550.0, gt=0, description="опорная длина волны, нм")

    # потери (мощностные)
    loss_db_cm: float = Field(0.0, ge=0, description="потери на распространение, дБ/см")

    # связь (амплитудные self-coupling)
    t1: float = Field(..., gt=0, lt=1, description="self-coupling первого/единственного коуплера")
    t2: float | None = Field(
        None, gt=0, lt=1, description="self-coupling второго коуплера (add-drop); None = all-pass"
    )

    @model_validator(mode="after")
    def _check(self) -> RingModelParams:
        if self.n_g0 < self.n_eff0:
            # для волноводов с нормальной дисперсией n_g > n_eff; предупреждаем явной ошибкой
            raise ValueError(
                f"n_g0 ({self.n_g0}) < n_eff0 ({self.n_eff0}): нефизично для типичного волновода"
            )
        return self

    # --- производные величины ---

    @property
    def round_trip_length_um(self) -> float:
        """Длина обхода: 2*pi*R + 2*L_c (для чистого кольца L_c = 0)."""
        return 2.0 * math.pi * self.ring_radius + 2.0 * self.coupling_length

    @property
    def amplitude_a(self) -> float:
        """Амплитудное пропускание за один обход: a = exp(-alpha_power * L / 2)."""
        alpha_per_um = self.loss_db_cm * _DB_PER_CM_TO_NP_PER_UM
        return math.exp(-0.5 * alpha_per_um * self.round_trip_length_um)

    def n_eff(self, lam_nm: np.ndarray | float) -> np.ndarray | float:
        """Линейная дисперсия фазового индекса вокруг lam0."""
        slope = (self.n_eff0 - self.n_g0) / self.lam0_nm  # dn_eff/dlam, 1/нм
        return self.n_eff0 + slope * (np.asarray(lam_nm, dtype=float) - self.lam0_nm)

    def phase(self, lam_nm: np.ndarray | float) -> np.ndarray | float:
        """Набег фазы за обход: phi = 2*pi*n_eff(lam)*L / lam (всё в согласованных единицах)."""
        lam = np.asarray(lam_nm, dtype=float)
        l_nm = self.round_trip_length_um * 1.0e3  # мкм -> нм
        return 2.0 * math.pi * self.n_eff(lam) * l_nm / lam


def all_pass_transmission(lam_nm: np.ndarray, p: RingModelParams) -> np.ndarray:
    """Пропускание through-порта all-pass кольца (мощность).

    T = (a^2 - 2 a t cos(phi) + t^2) / (1 - 2 a t cos(phi) + (a t)^2)
    """
    lam = np.asarray(lam_nm, dtype=float)
    a = p.amplitude_a
    t = p.t1
    cphi = np.cos(p.phase(lam))
    num = a * a - 2.0 * a * t * cphi + t * t
    den = 1.0 - 2.0 * a * t * cphi + (a * t) ** 2
    return num / den


def add_drop_transmission(lam_nm: np.ndarray, p: RingModelParams) -> tuple[np.ndarray, np.ndarray]:
    """Through- и drop-порты add-drop кольца (мощность).

    Требует p.t2. Возвращает (T_pass, T_drop).
    """
    if p.t2 is None:
        raise ValueError("add_drop_transmission требует t2 (второй коуплер)")
    lam = np.asarray(lam_nm, dtype=float)
    a = p.amplitude_a
    t1, t2 = p.t1, p.t2
    cphi = np.cos(p.phase(lam))
    den = 1.0 - 2.0 * t1 * t2 * a * cphi + (t1 * t2 * a) ** 2
    t_pass = (t2 * t2 * a * a - 2.0 * t1 * t2 * a * cphi + t1 * t1) / den
    t_drop = ((1.0 - t1 * t1) * (1.0 - t2 * t2) * a) / den
    return t_pass, t_drop


@dataclass
class FOM:
    """Figures of merit аналитической модели (на lam0)."""

    fsr_nm: float
    q_loaded: float
    q_intrinsic: float
    q_coupling: float
    t_min: float  # минимум through-порта в резонансе (для all-pass — глубина провала)
    extinction_ratio_db: float

    def report(self) -> str:
        return (
            f"FSR        = {self.fsr_nm:.3f} nm\n"
            f"Q_loaded   = {self.q_loaded:.0f}\n"
            f"Q_intrinsic= {self.q_intrinsic:.0f}\n"
            f"Q_coupling = {self.q_coupling:.0f}\n"
            f"T_min      = {self.t_min:.4f}\n"
            f"ER         = {self.extinction_ratio_db:.2f} dB"
        )


def figures_of_merit(p: RingModelParams) -> FOM:
    """FOM на опорной длине волны lam0 (через групповой индекс)."""
    a = p.amplitude_a
    l_nm = p.round_trip_length_um * 1.0e3
    lam0 = p.lam0_nm
    fsr_nm = lam0 * lam0 / (p.n_g0 * l_nm)

    # эффективный продукт пропускания: all-pass -> t1, add-drop -> t1*t2
    t_eff = p.t1 if p.t2 is None else p.t1 * p.t2

    def _q(ta: float) -> float:
        if ta >= 1.0:
            return math.inf
        return math.pi * p.n_g0 * l_nm * math.sqrt(ta) / (lam0 * (1.0 - ta))

    q_loaded = _q(t_eff * a)
    q_intrinsic = _q(a)  # связь идеальна (t -> 1): только потери
    inv_c = max(1.0 / q_loaded - 1.0 / q_intrinsic, 1e-30)
    q_coupling = 1.0 / inv_c

    t_min = ((a - t_eff) / (1.0 - a * t_eff)) ** 2
    er_db = -10.0 * math.log10(t_min) if t_min > 0 else math.inf
    return FOM(fsr_nm, q_loaded, q_intrinsic, q_coupling, t_min, er_db)
