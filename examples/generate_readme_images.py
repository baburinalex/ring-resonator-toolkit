"""Generate images for the project README.

Uses gdsfactory's built-in plotting for layouts and matplotlib for parameter sweeps.
Outputs are saved to examples/results/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ring_toolkit.components import ring_resonator_all_pass

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_layout_images() -> None:
    """Render each ring configuration as a standalone PNG via gdsfactory.plot()."""
    configs = [
        {"radius": 5.0, "gap": 0.15, "coupling_length": 0.0, "name": "ring_compact"},
        {"radius": 10.0, "gap": 0.2, "coupling_length": 0.0, "name": "ring_standard"},
        {"radius": 10.0, "gap": 0.2, "coupling_length": 8.0, "name": "ring_racetrack"},
    ]

    for cfg in configs:
        name = cfg.pop("name")
        component = ring_resonator_all_pass(**cfg)

        # gdsfactory's built-in plot returns a matplotlib Axes; grab its figure
        ax = component.plot()
        fig = ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf()

        output_path = OUTPUT_DIR / f"{name}.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close(fig)


def generate_parameter_sweep_preview() -> None:
    """Generate an illustrative Q-factor heatmap based on a simplified analytical model.

    The model captures the dominant physics of an all-pass ring resonator:
      Q_intrinsic combines propagation loss (constant in dB/cm) with bending
      loss that grows exponentially below a characteristic radius.
      Q_coupling derives from an exponentially decaying coupling coefficient,
      κ ∝ exp(-α·gap), giving Q_c ∝ exp(2·α·gap) / L_ring.
      Q_loaded follows 1/Q_loaded = 1/Q_intrinsic + 1/Q_coupling.

    Parameters are typical for a 220 nm SOI strip waveguide at 1550 nm.
    Full numerical accuracy still requires FDTD or coupled-mode solvers;
    this is for illustration of the sweep workflow.
    """
    # Sweep ranges
    radii = np.linspace(3, 30, 80)        # μm
    gaps = np.linspace(0.1, 0.5, 80)      # μm
    R, G = np.meshgrid(radii, gaps)

    # --- Intrinsic Q (propagation + bending loss) ---
    # Propagation loss: ~2 dB/cm typical for SOI strip @ 1550 nm
    alpha_prop_db_per_cm = 2.0
    # Convert to alpha in 1/μm: α [1/μm] = α[dB/cm] · ln(10)/10 · 1e-4
    alpha_prop = alpha_prop_db_per_cm * np.log(10) / 10 * 1e-4  # 1/μm

    # Bending loss: empirical exponential, becomes significant below ~5 μm
    R_crit = 4.0  # μm, characteristic bending-loss radius for SOI 500 nm strip
    alpha_bend = 0.5 * np.exp(-(R - R_crit) / 1.0)  # 1/μm, decays fast above R_crit

    alpha_total = alpha_prop + alpha_bend  # 1/μm, total propagation loss

    # Q_intrinsic = (2π·n_g) / (λ·α), with group index n_g ≈ 4.2 for SOI strip
    n_g = 4.2
    wavelength = 1.55  # μm
    Q_intrinsic = 2 * np.pi * n_g / (wavelength * alpha_total)

    # --- Coupling Q ---
    # κ² ∝ exp(-2·α_evan·gap); typical decay constant α_evan ~ 8 1/μm for SOI
    alpha_evan = 7.0  # 1/μm
    L_ring = 2 * np.pi * R  # ring round-trip length
    # Simplified: Q_coupling ∝ L_ring · exp(2·α_evan·gap)
    Q_coupling = (L_ring / wavelength) * np.exp(2 * alpha_evan * G)

    # --- Loaded Q ---
    Q_loaded = 1.0 / (1.0 / Q_intrinsic + 1.0 / Q_coupling)

    # Plot on log scale — Q spans several orders of magnitude
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(
        R, G, np.log10(Q_loaded),
        levels=20,
        cmap="viridis",
    )
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r"$\log_{10}(Q_{\mathrm{loaded}})$")

    # Overlay the critical-coupling contour where Q_intrinsic ≈ Q_coupling
    critical = ax.contour(
        R, G, Q_intrinsic - Q_coupling,
        levels=[0], colors="red", linewidths=2, linestyles="--",
    )
    ax.clabel(critical, fmt={0: "critical coupling"}, fontsize=9)

    ax.set_xlabel("Ring radius (μm)")
    ax.set_ylabel("Coupling gap (μm)")
    ax.set_title(
        "Loaded Q-factor over geometry parameter space\n"
        "(simplified model: SOI strip, λ = 1.55 μm)"
    )

    output_path = OUTPUT_DIR / "parameter_sweep_preview.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating README images...")
    generate_layout_images()
    generate_parameter_sweep_preview()
    print("Done.")