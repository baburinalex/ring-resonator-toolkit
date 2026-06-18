"""
Параметры моделей (валидируются pydantic).

Все длины — в МИКРОМЕТРАХ, как в исходных Lumerical-скриптах.
Методы ``doubles()`` / ``strings()`` отдают плоские словари переменных,
которые слой ``sim`` кладёт в рабочее поле Lumerical через
``putDouble`` / ``putString``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Platform(BaseModel):
    """Платформа: толщина плёнки и материалы сердцевины/обкладки.

    По умолчанию — SOI (кремниевый strip-волновод 220 нм в SiO2).
    Пресеты: ``Platform.soi()`` и ``Platform.sin()``.
    """

    wg_height: float = Field(0.22, gt=0, description="толщина плёнки, мкм")
    material_core: str = "Si (Silicon) - Palik"
    material_clad: str = "SiO2 (Glass) - Palik"

    @classmethod
    def soi(cls, wg_height: float = 0.22) -> Platform:
        """SOI: кремниевое ядро, оксидная обкладка (стандарт 220 нм)."""
        return cls(
            wg_height=wg_height,
            material_core="Si (Silicon) - Palik",
            material_clad="SiO2 (Glass) - Palik",
        )

    @classmethod
    def sin(cls, wg_height: float = 0.22) -> Platform:
        """Si3N4: нитрид кремния в оксиде."""
        return cls(
            wg_height=wg_height,
            material_core="Si3N4 (Silicon Nitride) - Luke",
            material_clad="SiO2 (Glass) - Palik",
        )


class SimParams(BaseModel):
    """Параметры FDTD-решателя и спектрального диапазона."""

    lam_start: float = Field(1500.0, gt=0, description="начало диапазона, нм")
    lam_stop: float = Field(1600.0, gt=0, description="конец диапазона, нм")
    freq_points: int = Field(2000, gt=1, description="точек по частоте")
    sim_time: float = Field(3000.0, gt=0, description="время моделирования, фс")
    mesh_override: float = Field(20.0, gt=0, description="ячейка в волноводах, нм (SOI: ~20)")
    mesh_accuracy: int = Field(2, ge=1, le=8, description="точность фоновой сетки")

    def model_post_init(self, _ctx) -> None:  # noqa: D401
        if self.lam_stop <= self.lam_start:
            raise ValueError("lam_stop должна быть больше lam_start")


class RingParams(BaseModel):
    """Кольцевой резонатор: all-pass (add_drop=0) или add-drop (add_drop=1)."""

    wg_width: float = Field(0.5, gt=0, description="ширина strip-волновода (SOI ~0.5), мкм")
    ring_radius: float = Field(5.0, gt=0, description="радиус по средней линии, мкм")
    gap: float = Field(0.2, gt=0, description="зазор край-в-край шина/кольцо, мкм")
    L_bus: float = Field(16.0, gt=0, description="длина шины вдоль x, мкм")
    add_drop: int = Field(0, ge=0, le=1, description="0=all-pass, 1=add-drop")

    platform: Platform = Field(default_factory=Platform)
    sim: SimParams = Field(default_factory=SimParams)

    def doubles(self) -> dict[str, float]:
        return {
            "wg_height": self.platform.wg_height,
            "wg_width": self.wg_width,
            "ring_radius": self.ring_radius,
            "gap": self.gap,
            "L_bus": self.L_bus,
            "add_drop": float(self.add_drop),
            "lam_start": self.sim.lam_start,
            "lam_stop": self.sim.lam_stop,
            "freq_points": float(self.sim.freq_points),
            "sim_time": self.sim.sim_time,
            "mesh_override": self.sim.mesh_override,
            "mesh_accuracy": float(self.sim.mesh_accuracy),
        }

    def strings(self) -> dict[str, str]:
        return {
            "material_core": self.platform.material_core,
            "material_clad": self.platform.material_clad,
        }


class MMIParams(BaseModel):
    """1x2 MMI-делитель.

    Дефолты — стартовая оценка для SOI (220 нм, TE, 1550 нм) из теории
    симметричной интерференции (см. lumfdtd.design.mmi_1x2_symmetric для
    W_mmi=3 мкм, n_eff~2.85): L_mmi ~ 8.8 мкм, выходы на ± W/4.
    Точную длину доводят свипом run_mmi_sweep — W_e и n_eff зависят от
    платформы и дисперсии. Вход тейперится 0.5 -> 1.0 мкм, чтобы снизить
    потери на стыке с областью MMI.
    """

    L_input: float = Field(5.0, gt=0, description="длина тейпера вход/выход, мкм")
    input_width: float = Field(0.5, gt=0, description="одномодовый вход SOI, мкм")
    taper_width: float = Field(1.0, gt=0, description="ширина у грани MMI, мкм")
    dist_btw_out_tapers: float = Field(1.5, gt=0, description="центр-в-центр выходов, мкм")
    W_mmi: float = Field(3.0, gt=0, description="ширина области MMI, мкм")
    L_mmi: float = Field(8.8, gt=0, description="длина области MMI, мкм (оценка, свипуется)")

    platform: Platform = Field(default_factory=Platform)

    def doubles(self) -> dict[str, float]:
        return {
            "wg_height": self.platform.wg_height,
            "L_input": self.L_input,
            "input_width": self.input_width,
            "taper_width": self.taper_width,
            "dist_btw_out_tapers": self.dist_btw_out_tapers,
            "W_mmi": self.W_mmi,
            "L_mmi": self.L_mmi,
        }

    def strings(self) -> dict[str, str]:
        return {
            "material_core": self.platform.material_core,
            "material_clad": self.platform.material_clad,
        }
