"""
Слой запуска расчётов через Ansys Lumerical lumapi.

lumapi импортируется ЛЕНИВО (внутри функций), поэтому пакет и тесты
импортируются и работают без установленного Lumerical. Для фактического
запуска FDTD нужна лицензия Lumerical и правильный путь к lumapi.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .geometry import draw_ring_geometry
from .params import RingParams

DEFAULT_LUMAPI_PATH = r"C:\Program Files\Lumerical\v241\api\python"


def _import_lumapi(path: str = DEFAULT_LUMAPI_PATH):
    if path and path not in sys.path:
        sys.path.append(path)
    try:
        import lumapi  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "Не найден lumapi. Установи Ansys Lumerical и/или поправь "
            f"lumapi_path (сейчас: {path!r})."
        ) from exc
    return lumapi


def _put_params(lumapi, h, params) -> None:
    for k, v in params.strings().items():
        lumapi.putString(h, k, v)
    for k, v in params.doubles().items():
        lumapi.putDouble(h, k, v)


# ---------------------------------------------------------------------
# RING: решатель + источник + мониторы + извлечение спектра
# ---------------------------------------------------------------------

_BUILD_RING_SIM_LSF = '''

# Центр кольца по y, в метрах (вход. переменные в мкм)
y_ring_m = (gap + wg_width + ring_radius) * 1e-6;

addfdtd;
set("dimension","3D");
set("x", 0);
set("x span", (L_bus + 2) * 1e-6);
set("y min", -(wg_width/2 + 1.5) * 1e-6);
set("y max", y_ring_m + (ring_radius + wg_width/2 + 1.0) * 1e-6);
set("z", 0);
set("z span", (wg_height + 2.4) * 1e-6);
set("background material", material_clad);
set("mesh accuracy", mesh_accuracy);
set("simulation time", sim_time * 1e-15);

addmesh;
set("name","mesh_override");
set("x", 0);
set("x span", (L_bus + 2) * 1e-6);
set("y min", -(wg_width/2 + 0.3) * 1e-6);
set("y max", y_ring_m + (ring_radius + wg_width/2 + 0.3) * 1e-6);
set("z", 0);
set("z span", (wg_height + 0.4) * 1e-6);
set("dx", mesh_override * 1e-9);
set("dy", mesh_override * 1e-9);
set("dz", (mesh_override * 0.5) * 1e-9);

addmode;
set("name","source");
set("injection axis","x-axis");
set("direction","Forward");
set("x", -(L_bus/2 - 1.0) * 1e-6);
set("y", 0);
set("y span", (wg_width + 2.0) * 1e-6);
set("z", 0);
set("z span", (wg_height + 1.6) * 1e-6);
set("mode selection","fundamental TE mode");

setglobalsource("wavelength start", lam_start * 1e-9);
setglobalsource("wavelength stop",  lam_stop  * 1e-9);
setglobalmonitor("frequency points", freq_points);
setglobalmonitor("use source limits", 1);

addpower;
set("name","in");
set("monitor type","2D X-normal");
set("x", -(L_bus/2 - 2.0) * 1e-6);
set("y", 0);
set("y span", (wg_width + 2.0) * 1e-6);
set("z", 0);
set("z span", (wg_height + 1.6) * 1e-6);

addpower;
set("name","through");
set("monitor type","2D X-normal");
set("x", (L_bus/2 - 1.5) * 1e-6);
set("y", 0);
set("y span", (wg_width + 2.0) * 1e-6);
set("z", 0);
set("z span", (wg_height + 1.6) * 1e-6);

if (add_drop == 1) {
    addpower;
    set("name","drop");
    set("monitor type","2D X-normal");
    set("x", -(L_bus/2 - 1.5) * 1e-6);
    set("y", 2 * y_ring_m);
    set("y span", (wg_width + 2.0) * 1e-6);
    set("z", 0);
    set("z span", (wg_height + 1.6) * 1e-6);
}
'''

_RUN_RING_LSF = '''
run;
T_in      = transmission("in");
T_through = transmission("through");
f         = getdata("through","f");
lam       = c / f;
'''


@dataclass
class RingSpectrum:
    lam_nm: np.ndarray
    t_through: np.ndarray
    t_in: np.ndarray

    @property
    def t_norm(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(self.t_in) > 1e-9, self.t_through / self.t_in, self.t_through)


def run_ring(
    params: RingParams | None = None,
    lumapi_path: str = DEFAULT_LUMAPI_PATH,
    build_only: bool = False,
    hide: bool = False,
) -> RingSpectrum | None:
    """Построить и (если build_only=False) посчитать кольцевой резонатор.

    build_only=True: модель строится, но не считается — удобно открыть
    .fsp в GUI и проверить геометрию/сетку перед долгим расчётом.
    """
    params = params or RingParams()
    lumapi = _import_lumapi(lumapi_path)
    h = lumapi.open("fdtd", hide=hide)

    _put_params(lumapi, h, params)
    lumapi.evalScript(h, draw_ring_geometry())
    lumapi.evalScript(h, _BUILD_RING_SIM_LSF)

    if build_only:
        return None

    lumapi.evalScript(h, _RUN_RING_LSF)
    lam_nm = np.asarray(lumapi.getVar(h, "lam")).ravel() * 1e9
    t_through = np.asarray(lumapi.getVar(h, "T_through")).ravel()
    t_in = np.asarray(lumapi.getVar(h, "T_in")).ravel()
    return RingSpectrum(lam_nm=lam_nm, t_through=t_through, t_in=t_in)

