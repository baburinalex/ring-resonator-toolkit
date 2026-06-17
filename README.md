# lumerical-fdtd-photonics

[![CI](https://github.com/baburinalex/lumerical-fdtd-photonics/actions/workflows/ci.yml/badge.svg)](https://github.com/baburinalex/lumerical-fdtd-photonics/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

FDTD modeling of **silicon-on-insulator (SOI)** photonic components in
**Ansys Lumerical**, driven from Python via `lumapi` (the platform is
configurable — `Platform.soi()` / `Platform.sin()` for Si₃N₄):

- **1×2 MMI splitter** — loss sweep over the multimode-section length/width.
- **Ring resonator** — all-pass (notch) or add-drop, with through-port
  transmission `T(λ)` and extraction of **FSR** and loaded **Q**.

The codebase separates three concerns: validated **parameters** (pydantic),
**geometry** (Lumerical `lsf` script generators), and **simulation** (`lumapi`
driver). Spectrum analysis is pure NumPy and fully unit-tested — so **CI runs
green without a Lumerical license**.

## Why the structure

A naive full-3D-FDTD of a ring is expensive *by physics*: resolving a resonance
of quality factor `Q` requires simulating on the order of the photon lifetime
`τ ≈ Q·λ / (2πc)` (tens of ps for `Q ≈ 10⁴` at 1550 nm) on a fine mesh covering
the whole ring. On SOI the high index contrast makes compact rings practical, so
the default `R = 5 µm` is a **realistic** size that also completes quickly; bear
in mind that high-index silicon needs a finer mesh (default ~20 nm).

For high-Q rings the recommended path is hybrid and much cheaper: extract the
field coupling `κ`, `t` from an FDTD/EME simulation of the **coupling region
only**, get `n_g` and propagation loss from a mode/EME solve, then assemble the
all-pass / add-drop transfer function analytically (coupled-mode theory). See the
roadmap.

## Install

```bash
git clone https://github.com/baburinalex/lumerical-fdtd-photonics.git
cd lumerical-fdtd-photonics
pip install -e ".[dev]"          # add ,plot for matplotlib in examples
```

`lumapi` ships with the Ansys Lumerical installation (not on PyPI). Point the
code at it via the `lumapi_path` argument or edit the default in
`src/lumfdtd/sim.py`.

## Quickstart — ring resonator

```python
from lumfdtd.params import Platform, RingParams, SimParams
from lumfdtd.sim import run_ring
from lumfdtd.analysis import analyze_spectrum

params = RingParams(
    wg_width=0.5, ring_radius=5.0, gap=0.2, L_bus=16.0, add_drop=0,
    platform=Platform.soi(),                       # SOI is the default
    sim=SimParams(lam_start=1500, lam_stop=1600,
                  freq_points=2000, sim_time=3000, mesh_override=20),
)

# build_only=True first to inspect geometry/mesh in the Lumerical GUI
spec = run_ring(params, build_only=False)
print(analyze_spectrum(spec.lam_nm, spec.t_norm).report())
```

Runnable script: [`examples/run_ring.py`](examples/run_ring.py)
(saves `ring_transmission.txt` and a `.png`).

## Quickstart — MMI loss sweep (SOI)

Default 1×2 dimensions are a starting estimate from symmetric-interference
self-imaging theory for 220 nm SOI; the sweep refines `L_mmi`:

```python
import numpy as np
from lumfdtd.params import MMIParams, Platform
from lumfdtd.design import mmi_1x2_symmetric, suggest_length_sweep_um
from lumfdtd.sim import run_mmi_sweep

est = mmi_1x2_symmetric(3.0, n_eff=2.85)        # L_pi, L_mmi, output spacing
print(est)                                       # ~ L_mmi 8.3 µm, outputs ±W/4

losses = run_mmi_sweep(
    base_fsp=r"C:\path\to\mmi_fine_mesh.fsp",    # FDTD + monitors in/out_2/out_3
    l_mmi_values=suggest_length_sweep_um(est.L_mmi_um),
    w_mmi_values=[2.75, 3.0, 3.25],
    params=MMIParams(W_mmi=3.0, dist_btw_out_tapers=est.out_sep_um,
                     platform=Platform.soi()),
)
```

`L_π = 4·n_eff·W_e² / (3λ)`, and for a centered-input 1×2 splitter
`L_mmi = 3·L_π/8` with outputs at `±W_e/4`. The analytic value is only a
starting point — `W_e` and `n_eff` are platform/dispersion dependent, so the
FDTD sweep finds the real optimum.

Runnable script: [`examples/run_mmi_sweep.py`](examples/run_mmi_sweep.py).

## Layout

```
src/lumfdtd/
  params.py      # pydantic: Platform, SimParams, RingParams, MMIParams
  geometry.py    # draw_ring_geometry(), draw_mmi_geometry()  -> lsf strings
  sim.py         # run_ring(), run_mmi_sweep()  (lazy lumapi import)
  analysis.py    # find_resonances(), estimate_q(), analyze_spectrum()  (NumPy)
  design.py      # mmi_1x2_symmetric(): analytic 1x2 MMI geometry estimate
examples/        # run_ring.py, run_mmi_sweep.py
tests/           # pytest, no Lumerical needed
```

## Testing & lint

```bash
pytest -q
ruff check .
```

## Roadmap

- Hybrid ring model: FDTD/EME coupler → (`κ`, `t`) → analytical all-pass /
  add-drop transfer function with propagation loss (cheap, physically explicit;
  pairs with `ring-resonator-toolkit`).
- 2.5D varFDTD (Lumerical MODE) backend for full-size rings.
- z-symmetry boundary in FDTD for a free 2× speedup (mode-parity dependent).

## License

MIT — see [LICENSE](LICENSE).
