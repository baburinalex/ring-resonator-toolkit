"""
Анализ спектра пропускания кольцевого резонатора (только numpy).

Резонансы all-pass кольца — провалы (dips) в T(lambda). Модуль ищет провалы
ПО СЫРЫМ данным (без сглаживания, которое размывает узкие резонансы),
отсекает мелкую рябь порогом по глубине, оценивает нагруженную Q по ширине
на полувысоте (FWHM) и считает FSR и групповой индекс n_g.

Окна анализа задаются в НАНОМЕТРАХ (а не в точках), поэтому модуль одинаково
корректен и на мелкой сетке по длине волны, и на крупной.

Не зависит от lumapi/Lumerical -> полностью покрывается юнит-тестами и работает
в CI без лицензии.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Resonance:
    lambda0_nm: float
    q: float
    depth: float


@dataclass
class SpectrumAnalysis:
    resonances: list[Resonance] = field(default_factory=list)
    fsr_nm: list[float] = field(default_factory=list)
    n_g: float = float("nan")

    @property
    def mean_fsr_nm(self) -> float:
        return float(np.mean(self.fsr_nm)) if self.fsr_nm else float("nan")

    @property
    def mean_q(self) -> float:
        qs = [r.q for r in self.resonances if np.isfinite(r.q)]
        return float(np.mean(qs)) if qs else float("nan")

    def report(self) -> str:
        lines = ["--- Анализ спектра through-порта ---"]
        if not self.resonances:
            lines.append("Резонансы не найдены.")
            return "\n".join(lines)
        for r in self.resonances:
            q = f"{r.q:8.0f}" if np.isfinite(r.q) else "    n/a"
            lines.append(
                f"  lambda0 = {r.lambda0_nm:8.2f} нм | Q = {q} | глубина = {r.depth:.3f}"
            )
        if self.fsr_nm:
            fsr = ", ".join(f"{v:.2f}" for v in self.fsr_nm)
            lines.append(f"  FSR, нм: {fsr}  (средний {self.mean_fsr_nm:.2f})")
        if np.isfinite(self.mean_q):
            lines.append(f"  средняя нагруженная Q = {self.mean_q:.0f}")
        if np.isfinite(self.n_g):
            lines.append(f"  групповой индекс n_g = {self.n_g:.2f}")
        return "\n".join(lines)


def _win_points(lam_nm: np.ndarray, win_nm: float) -> int:
    """Перевод окна из нанометров в число точек по фактическому шагу сетки."""
    n = len(lam_nm)
    if n < 2:
        return 1
    step = abs(lam_nm[-1] - lam_nm[0]) / (n - 1)
    return max(1, int(round(win_nm / step)))


def find_resonances(
    lam_nm: np.ndarray,
    t: np.ndarray,
    min_depth: float = 0.15,
    baseline_win_nm: float = 6.0,
    min_sep_nm: float = 1.0,
) -> list[int]:
    """
    Индексы провалов по СЫРЫМ данным.

    Провал = локальный минимум, глубина которого (локальная база минус значение)
    >= min_depth. Локальная база — максимум T в окне +-baseline_win_nm, что
    отсекает мелкую рябь (её глубина мала). Близкие минимумы (< min_sep_nm)
    схлопываются в самый глубокий, чтобы рябь на дне не дробила провал.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    w = _win_points(lam_nm, baseline_win_nm)
    cand = []
    for i in range(1, n - 1):
        if t[i] <= t[i - 1] and t[i] < t[i + 1]:
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            baseline = float(np.max(t[lo:hi]))
            if (baseline - t[i]) >= min_depth:
                cand.append(i)
    if not cand:
        return []
    groups: list[list[int]] = [[cand[0]]]
    for i in cand[1:]:
        if abs(lam_nm[i] - lam_nm[groups[-1][-1]]) <= min_sep_nm:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(g[int(np.argmin(t[g]))]) for g in groups]


def estimate_q(
    lam_nm: np.ndarray, t: np.ndarray, idx: int, win_nm: float = 5.0
) -> tuple[float, float, float]:
    """
    Нагруженная Q из ширины провала на полувысоте (FWHM) по сырым данным.
    Окно win_nm в нанометрах (должно быть меньше FSR, но шире провала).
    Возвращает (lambda0, Q, depth).
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    w = _win_points(lam_nm, win_nm)
    lo = max(0, idx - w)
    hi = min(n, idx + w + 1)
    baseline = float(np.max(t[lo:hi]))
    tmin = float(t[idx])
    depth = baseline - tmin
    lam0 = float(lam_nm[idx])
    half = 0.5 * (baseline + tmin)

    def _cross(rng) -> float | None:
        prev = None
        for i in rng:
            if prev is not None:
                y0, y1 = t[prev], t[i]
                if (y0 - half) * (y1 - half) <= 0 and y1 != y0:
                    x0, x1 = lam_nm[prev], lam_nm[i]
                    return x0 + (half - y0) * (x1 - x0) / (y1 - y0)
            prev = i
        return None

    left = _cross(range(idx, lo - 1, -1))
    right = _cross(range(idx, hi))
    if left is None or right is None:
        return lam0, float("nan"), depth
    fwhm = abs(right - left)
    if fwhm <= 0:
        return lam0, float("nan"), depth
    return lam0, lam0 / fwhm, depth


def group_index_from_fsr(
    fsr_nm: float, lambda0_nm: float, radius_um: float, coupling_length_um: float = 0.0
) -> float:
    """
    Групповой индекс из FSR кольца:  FSR = lambda^2 / (n_g * L),  L = 2*pi*R + 2*Lc.
    Длины волн в нм, радиус/длина связи в мкм. Возвращает n_g.
    """
    if fsr_nm <= 0 or radius_um <= 0:
        return float("nan")
    lam_m = lambda0_nm * 1e-9
    fsr_m = fsr_nm * 1e-9
    length_m = (2.0 * np.pi * radius_um + 2.0 * coupling_length_um) * 1e-6
    return float(lam_m**2 / (fsr_m * length_m))


def analyze_spectrum(
    lam_nm: np.ndarray,
    t: np.ndarray,
    min_depth: float = 0.15,
    radius_um: float | None = None,
    coupling_length_um: float = 0.0,
) -> SpectrumAnalysis:
    """
    Полный разбор: резонансы (по сырым данным, порог глубины) + Q + FSR.
    Если задан radius_um — дополнительно считает n_g из среднего FSR.
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(lam_nm)
    lam_nm, t = lam_nm[order], t[order]

    dips = find_resonances(lam_nm, t, min_depth=min_depth)
    res = [Resonance(*estimate_q(lam_nm, t, i)) for i in dips]

    centers = np.array(sorted(r.lambda0_nm for r in res))
    fsr = [float(v) for v in np.diff(centers)] if len(centers) >= 2 else []

    out = SpectrumAnalysis(resonances=res, fsr_nm=fsr)
    if radius_um is not None and fsr:
        lam0 = float(np.mean(centers))
        out.n_g = group_index_from_fsr(out.mean_fsr_nm, lam0, radius_um, coupling_length_um)
    return out
