"""
Пример: расчёт кольцевого резонатора + анализ спектра (FSR, Q).

Запускать ТОЛЬКО в терминале (не через кнопку Debug VS Code), из среды,
где доступен lumapi.

Совет по шагам:
  1) поставь BUILD_ONLY = True, проверь геометрию в GUI Lumerical;
  2) затем BUILD_ONLY = False и запусти расчёт.
"""

import os

from lumfdtd.analysis import analyze_spectrum
from lumfdtd.params import Platform, RingParams, SimParams
from lumfdtd.sim import run_ring

BUILD_ONLY = False
LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    params = RingParams(
        wg_width=0.5,          # одномодовый strip-волновод SOI
        ring_radius=5.0,       # для SOI это вполне реалистичный радиус
        gap=0.2,
        L_bus=16.0,
        add_drop=0,
        platform=Platform.soi(),   # кремний-на-изоляторе, 220 нм
        sim=SimParams(
            lam_start=1500.0, lam_stop=1600.0,
            freq_points=2000, sim_time=3000.0, mesh_override=20.0,
        ),
    )

    spec = run_ring(params, lumapi_path=LUMAPI_PATH, build_only=BUILD_ONLY)
    if spec is None:
        print("BUILD_ONLY=True: модель построена, расчёт не запускался.")
        return

    result = analyze_spectrum(spec.lam_nm, spec.t_norm)
    print(result.report())

    txt = os.path.join(OUTPUT_DIR, "ring_transmission.txt")
    import numpy as np

    np.savetxt(
        txt,
        np.column_stack([spec.lam_nm, spec.t_norm, spec.t_in]),
        header="lambda_nm  T_through_norm  T_in",
        comments="",
    )
    print(f"Спектр сохранён: {txt}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        plt.plot(spec.lam_nm, spec.t_norm, lw=1.2)
        plt.xlabel("Длина волны, нм")
        plt.ylabel("Пропускание through-порта")
        plt.title(
            f"Ring: R={params.ring_radius} мкм, "
            f"gap={params.gap} мкм, w={params.wg_width} мкм"
        )
        plt.grid(True, alpha=0.3)
        png = os.path.join(OUTPUT_DIR, "ring_transmission.png")
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        print(f"График сохранён: {png}")
    except Exception as exc:  # noqa: BLE001
        print(f"(График не построен: {exc})")


if __name__ == "__main__":
    main()
