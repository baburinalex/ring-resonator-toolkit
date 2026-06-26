"""
Внутренняя добротность Q_i от потерь для кольца из нитрида кремния (Si3N4), R = 1000 мкм.

Q_i = figures_of_merit(...).q_intrinsic (добротность при идеальной связи: только потери).

Физика: при низких потерях Q_i = 2*pi*n_g/(lam*alpha) — зависит только от n_g и потерь
на единицу длины, НЕ от радиуса. Поэтому большое кольцо R=300 даёт ту же кривую Q_i(loss),
что и малое; радиус влияет лишь при больших потерях (заметный набег alpha*L за обход).
У SiN малый контраст индекса (низкий n_g ~2), но очень низкие потери, отсюда — очень высокие Q_i.

Запуск (venv активен):
    python examples/qi_vs_loss_sin.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ring_toolkit.analytical_model import RingModelParams, figures_of_merit

# Si3N4-кольцо; n_eff0/n_g0 — типовые значения, уточни своим мод-солвером.
# t1 на Q_i не влияет (intrinsic = только потери).
RING = dict(ring_radius=1000.0, n_eff0=1.70, n_g0=2.05, lam0_nm=1550.0, t1=0.99)

# опорная точка для метки (хорошая SiN-платформа ~0.04 дБ/см)
OPERATING_LOSS_DB_CM = 0.04


def q_intrinsic(loss_db_cm: float) -> float:
    p = RingModelParams(loss_db_cm=loss_db_cm, **RING)
    return figures_of_merit(p).q_intrinsic


def main() -> None:
    loss = np.logspace(-3, 1, 250)  # 0.001 .. 10 дБ/см (режим SiN)
    qi = np.array([q_intrinsic(x) for x in loss])

    q_op = q_intrinsic(OPERATING_LOSS_DB_CM)
    print(f"Si3N4 ring: R = {RING['ring_radius']} um, n_g = {RING['n_g0']}, "
          f"lam0 = {RING['lam0_nm']} nm")
    for x in (0.01, 0.1, 1.0, 10.0):
        print(f"  loss = {x:6.3f} dB/cm  ->  Q_i = {q_intrinsic(x):,.0f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен — график пропущен")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(loss, qi, color="#2da44e", lw=2.2, label=r"$Q_i$ (analytical, Si$_3$N$_4$)")

    ax.scatter([OPERATING_LOSS_DB_CM], [q_op], color="#d1242f", zorder=5)
    ax.annotate(
        f"{OPERATING_LOSS_DB_CM:g} dB/cm,  $Q_i$≈{q_op:,.0f}",
        xy=(OPERATING_LOSS_DB_CM, q_op),
        xytext=(OPERATING_LOSS_DB_CM * 3, q_op * 0.3),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#d1242f", lw=1.2),
    )

    ref = qi[0] * loss[0]
    ax.loglog(loss, ref / loss, color="#8b949e", ls="--", lw=1.0,
              label=r"наклон $-1$:  $Q_i \propto 1/\alpha$")

    ax.set_xlabel("propagation loss (dB/cm)")
    ax.set_ylabel("intrinsic quality factor  $Q_i$")
    ax.set_title(rf"Intrinsic Q vs loss  (Si$_3$N$_4$ ring, R = {RING['ring_radius']:g} µm)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(__file__).parent / "results" / "qi_vs_loss_sin.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nграфик сохранён: {out}")


if __name__ == "__main__":
    main()
