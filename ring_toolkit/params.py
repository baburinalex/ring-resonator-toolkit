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
    Пресет: ``Platform.soi()``.
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
    L_bus: float = Field(22.0, gt=0, description="длина шины вдоль x, мкм (длиннее кольца)")
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

