"""
Пример: свип потерь 1x2 MMI-делителя на SOI по длине области MMI.
Стартовая длина берётся из аналитической оценки (симметричная
интерференция), свип идёт вокруг неё. Загружается базовый .fsp с
настроенными FDTD/источником/мониторами in / out_2 / out_3.

Запускать в терминале из среды с lumapi.
"""

import os

import numpy as np

from lumfdtd.design import mmi_1x2_symmetric, suggest_length_sweep_um
from lumfdtd.params import MMIParams, Platform
from lumfdtd.sim import run_mmi_sweep

LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"
BASE_FSP = r"C:\Users\ba\Documents\lum\mmi\soi\mmi_fine_mesh.fsp"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    w_mmi = 3.0

    # Аналитическая оценка длины и расстояния между выходами (SOI, n_eff~2.85)
    est = mmi_1x2_symmetric(w_mmi, wavelength_um=1.55, n_eff=2.85)
    print(est)

    params = MMIParams(
        L_input=5.0,
        input_width=0.5,                       # одномодовый вход SOI
        taper_width=1.0,                       # расширение у грани MMI
        dist_btw_out_tapers=est.out_sep_um,    # ~ W/2
        W_mmi=w_mmi,
        L_mmi=est.L_mmi_um,
        platform=Platform.soi(),
    )

    # Свип длины вокруг аналитической оценки + слегка варьируем ширину
    l_mmi_values = suggest_length_sweep_um(est.L_mmi_um, span_frac=0.35, n=15)
    w_mmi_values = [w_mmi - 0.25, w_mmi, w_mmi + 0.25]

    losses = run_mmi_sweep(
        base_fsp=BASE_FSP,
        l_mmi_values=l_mmi_values,
        w_mmi_values=w_mmi_values,
        params=params,
        lumapi_path=LUMAPI_PATH,
    )

    out = os.path.join(OUTPUT_DIR, "mmi_losses.txt")
    np.savetxt(out, losses)
    print(f"Матрица потерь (дБ) сохранена: {out}")
    print(f"Лучшие потери: {losses.max():.4f} дБ (ближе к 0 = лучше)")


if __name__ == "__main__":
    main()
