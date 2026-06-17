"""
Анализ спектра пропускания кольцевого резонатора (только numpy).

Резонансы all-pass кольца — это провалы (dips) в T(lambda).
Модуль ищет провалы, оценивает нагруженную добротность Q по ширине
на полувысоте (FWHM) и считает свободный спектральный диапазон (FSR).

Важно: модуль не зависит от lumapi/Lumerical — поэтому он полностью
покрывается юнит-тестами и работает в CI без лицензии.
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

    @property
    def mean_fsr_nm(self) -> float:
        return float(np.mean(self.fsr_nm)) if self.fsr_nm else float("nan")

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
        return "\n".join(lines)


def smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    """Простое скользящее среднее (для устойчивого поиска экстремумов)."""
    y = np.asarray(y, dtype=float)
    if k < 2 or len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="same")


def find_resonances(
    lam_nm: np.ndarray, t: np.ndarray, min_prominence_frac: float = 0.03
) -> list[int]:
    """Индексы провалов (локальных минимумов) с минимальной заметностью."""
    lam_nm = np.asarray(lam_nm, dtype=float)
    ts = smooth(np.asarray(t, dtype=float), 5)
    tmax = float(np.max(ts))
    dips: list[int] = []
    for i in range(1, len(ts) - 1):
        if ts[i] <= ts[i - 1] and ts[i] < ts[i + 1]:
            if (tmax - ts[i]) > min_prominence_frac * tmax:
                dips.append(i)
    return dips


def estimate_q(
    lam_nm: np.ndarray, t: np.ndarray, idx: int, win: int = 200
) -> tuple[float, float, float]:
    """
    Нагруженная Q из ширины провала на полувысоте (линейная интерполяция
    пересечений уровня полу-глубины). Возвращает (lambda0, Q, depth).
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    lo = max(0, idx - win)
    hi = min(n, idx + win + 1)
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


def analyze_spectrum(
    lam_nm: np.ndarray, t: np.ndarray, min_prominence_frac: float = 0.03
) -> SpectrumAnalysis:
    """Полный разбор: резонансы + Q + FSR."""
    lam_nm = np.asarray(lam_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    order = np.argsort(lam_nm)
    lam_nm, t = lam_nm[order], t[order]

    dips = find_resonances(lam_nm, t, min_prominence_frac)
    res = [Resonance(*estimate_q(lam_nm, t, i)) for i in dips]

    centers = np.array([r.lambda0_nm for r in res])
    fsr = list(np.diff(np.sort(centers))) if len(centers) >= 2 else []
    return SpectrumAnalysis(resonances=res, fsr_nm=[float(v) for v in fsr])
