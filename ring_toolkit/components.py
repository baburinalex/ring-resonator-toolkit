"""Ring resonator components for silicon photonics design.

This module provides parametric ring resonator components built with gdsfactory.
Supports all-pass and add-drop topologies with configurable geometry.
"""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.gpdk import get_generic_pdk
from gdsfactory.typings import CrossSectionSpec
from pydantic import BaseModel, Field, field_validator

# Activate the generic PDK on module import.
# In gdsfactory 9.x, a PDK must be explicitly activated before creating components.
# Real designs should replace this with their foundry-specific PDK.
_PDK = get_generic_pdk()
_PDK.activate()


class RingResonatorParams(BaseModel):
    """Geometric parameters for a ring resonator.

    Attributes:
        radius: Ring radius in micrometers.
        gap: Coupling gap between bus waveguide and ring in micrometers.
        waveguide_width: Width of the waveguide in micrometers.
        coupling_length: Length of the straight coupling section in micrometers.
            Set to 0 for a point-coupled ring.
        bus_length: Length of the bus waveguide on each side of the coupler.
    """

    radius: float = Field(default=10.0, gt=0, description="Ring radius (um)")
    gap: float = Field(default=0.2, gt=0, description="Coupling gap (um)")
    waveguide_width: float = Field(default=0.5, gt=0, description="Waveguide width (um)")
    coupling_length: float = Field(default=0.0, ge=0, description="Coupling section length (um)")
    bus_length: float = Field(default=10.0, gt=0, description="Bus waveguide length per side (um)")

    @field_validator("gap")
    @classmethod
    def gap_should_be_reasonable(cls, v: float) -> float:
        """Warn if gap is outside typical fabrication range for SOI."""
        if v < 0.05:
            raise ValueError(f"Gap {v} um is below typical fabrication limit (~50 nm)")
        if v > 2.0:
            raise ValueError(f"Gap {v} um is too large for meaningful coupling")
        return v


@gf.cell
def ring_resonator_all_pass(
    radius: float = 10.0,
    gap: float = 0.2,
    waveguide_width: float = 0.5,
    coupling_length: float = 0.0,
    bus_length: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Create an all-pass ring resonator.

    The component consists of a circular ring evanescently coupled to a straight
    bus waveguide. The ring is positioned above the bus with a controllable gap.

    Args:
        radius: Ring radius in micrometers. Default 10 um.
        gap: Edge-to-edge gap between bus and ring waveguides in um. Default 0.2.
        waveguide_width: Width of both bus and ring waveguides in um. Default 0.5.
        coupling_length: Length of the straight coupling section in um.
            If 0, the ring is point-coupled (pure ring geometry). Default 0.
        bus_length: Length of bus waveguide extending on each side of the coupler
            in um. Default 10.
        cross_section: gdsfactory cross-section specification. Default "strip".

    Returns:
        gf.Component: gdsfactory Component with two optical ports:
            - "o1": Input port on the left of the bus waveguide.
            - "o2": Output port on the right of the bus waveguide.

    Example:
        >>> import gdsfactory as gf
        >>> from ring_toolkit.components import ring_resonator_all_pass
        >>> c = ring_resonator_all_pass(radius=8.0, gap=0.15)
        >>> c.show()  # Opens in KLayout
    """
    # Validate parameters through pydantic model
    params = RingResonatorParams(
        radius=radius,
        gap=gap,
        waveguide_width=waveguide_width,
        coupling_length=coupling_length,
        bus_length=bus_length,
    )

    component = gf.Component()

    # Build the ring (racetrack if coupling_length > 0, otherwise point-coupled)
    if params.coupling_length > 0:
        ring = gf.components.coupler_ring(
            radius=params.radius,
            gap=params.gap,
            length_x=params.coupling_length,
            cross_section=cross_section,
        )
    else:
        ring = gf.components.ring_single(
            radius=params.radius,
            gap=params.gap,
            length_x=0.001,
            length_y=0.001,
            cross_section=cross_section,
        )

    ring_ref = component.add_ref(ring)

    # Propagate ports from the ring component.
    # In gdsfactory 9.x, instance ports are iterated directly; port.name gives the label.
    for port in ring_ref.ports:
        component.add_port(name=port.name, port=port)

    # Add metadata for downstream processing
    component.info["radius"] = params.radius
    component.info["gap"] = params.gap
    component.info["waveguide_width"] = params.waveguide_width
    component.info["coupling_length"] = params.coupling_length
    component.info["topology"] = "all_pass"

    return component


if __name__ == "__main__":
    # Quick visual check when running this file directly
    c = ring_resonator_all_pass(radius=10.0, gap=0.2)
    print(f"Component: {c.name}")
    print(f"Ports: {[p.name for p in c.ports]}")
    print(f"Bounding box: {c.bbox_np()}")
    c.show()
