"""Tests for ring resonator components."""

from __future__ import annotations

import gdsfactory as gf
import numpy as np
import pytest
from pydantic import ValidationError

from ring_toolkit.components import RingResonatorParams, ring_resonator_all_pass


class TestRingResonatorParams:
    """Test parameter validation via the pydantic model."""

    def test_default_parameters_are_valid(self):
        """Default parameters should construct without errors."""
        params = RingResonatorParams()
        assert params.radius == 10.0
        assert params.gap == 0.2
        assert params.waveguide_width == 0.5

    def test_negative_radius_rejected(self):
        """Radius must be positive."""
        with pytest.raises(ValidationError):
            RingResonatorParams(radius=-5.0)

    def test_zero_radius_rejected(self):
        """Radius must be strictly positive (gt=0)."""
        with pytest.raises(ValidationError):
            RingResonatorParams(radius=0.0)

    def test_gap_below_fab_limit_rejected(self):
        """Gap below 50 nm is unrealistic for SOI fabrication."""
        with pytest.raises(ValidationError, match="below typical fabrication limit"):
            RingResonatorParams(gap=0.01)

    def test_gap_too_large_rejected(self):
        """Gap above 2 um produces no meaningful coupling."""
        with pytest.raises(ValidationError, match="too large for meaningful coupling"):
            RingResonatorParams(gap=5.0)

    def test_typical_soi_parameters_accepted(self):
        """Standard SOI ring parameters should pass."""
        params = RingResonatorParams(
            radius=10.0,
            gap=0.15,
            waveguide_width=0.5,
        )
        assert params.gap == 0.15


class TestRingResonatorAllPass:
    """Test the all-pass ring resonator component generation."""

    def test_default_component_builds(self):
        """Component should build with default parameters."""
        c = ring_resonator_all_pass()
        assert isinstance(c, gf.Component)

    def test_component_has_two_optical_ports(self):
        """All-pass ring should expose exactly two ports for the bus waveguide."""
        c = ring_resonator_all_pass()
        port_names = [p.name for p in c.ports]
        assert len(port_names) == 2
        assert "o1" in port_names
        assert "o2" in port_names

    def test_metadata_stored_in_component_info(self):
        """Parameters should be retrievable from component.info for downstream tools."""
        c = ring_resonator_all_pass(radius=8.0, gap=0.18)
        assert c.info["radius"] == 8.0
        assert c.info["gap"] == 0.18
        assert c.info["topology"] == "all_pass"

    @pytest.mark.parametrize("radius", [5.0, 10.0, 20.0, 50.0])
    def test_bounding_box_scales_with_radius(self, radius):
        """The bounding box should grow proportionally with ring radius."""
        c = ring_resonator_all_pass(radius=radius)
        bbox = c.bbox_np()
        height = bbox[1][1] - bbox[0][1]
        # Height should be at least the ring diameter (2*radius)
        # plus waveguide width and gap contributions
        assert height >= 2 * radius - 1.0, (
            f"Bounding box height {height} too small for radius {radius}"
        )

    @pytest.mark.parametrize("gap", [0.1, 0.15, 0.2, 0.3, 0.5])
    def test_various_gaps_build_successfully(self, gap):
        """Component should build for a range of typical coupling gaps."""
        c = ring_resonator_all_pass(gap=gap)
        assert c.info["gap"] == gap

    def test_invalid_gap_raises_at_component_level(self):
        """Invalid gap should propagate as ValidationError when calling the factory."""
        with pytest.raises(ValidationError):
            ring_resonator_all_pass(gap=0.001)

    def test_different_parameters_produce_different_components(self):
        """gdsfactory should cache components per unique parameter set."""
        c1 = ring_resonator_all_pass(radius=10.0)
        c2 = ring_resonator_all_pass(radius=15.0)
        assert c1.name != c2.name

    def test_same_parameters_produce_cached_component(self):
        """Calling with the same parameters should return the cached cell."""
        c1 = ring_resonator_all_pass(radius=10.0, gap=0.2)
        c2 = ring_resonator_all_pass(radius=10.0, gap=0.2)
        assert c1.name == c2.name


class TestRacetrackGeometry:
    """Test ring with non-zero coupling length (racetrack topology)."""

    def test_racetrack_builds(self):
        """Ring with coupling_length > 0 should build as a racetrack."""
        c = ring_resonator_all_pass(radius=10.0, gap=0.2, coupling_length=5.0)
        assert isinstance(c, gf.Component)
        assert c.info["coupling_length"] == 5.0

    def test_racetrack_wider_than_pure_ring(self):
        """Racetrack should have a larger x-extent than a point-coupled ring of equal radius."""
        ring = ring_resonator_all_pass(radius=10.0, gap=0.2, coupling_length=0.0)
        racetrack = ring_resonator_all_pass(radius=10.0, gap=0.2, coupling_length=10.0)

        ring_width = ring.bbox_np()[1][0] - ring.bbox_np()[0][0]
        racetrack_width = racetrack.bbox_np()[1][0] - racetrack.bbox_np()[0][0]

        assert racetrack_width > ring_width
        # Racetrack should be wider by approximately the coupling length
        assert racetrack_width - ring_width == pytest.approx(10.0, abs=1.0)