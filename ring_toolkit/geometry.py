"""
Генераторы Lumerical lsf-скриптов геометрии.

Скрипты обращаются к переменным рабочего поля Lumerical
(``wg_width``, ``ring_radius`` и т.д. в мкм; ``material_core`` — строка),
которые слой ``sim`` заранее кладёт через ``putDouble`` / ``putString``.
Геометрия намеренно отделена от числовых значений, что позволяет удобно перерисовывать её в свипах
(switchtolayout -> delete -> redraw).
"""

from __future__ import annotations

_RING_GEOMETRY_LSF = '''

addstructuregroup;
set("name","ring coupler");
set("x", 0);
set("y", 0);
set("z", 0);

# Пользовательские свойства группы (2 = длина в метрах, 0 = число, 1 = строка)
adduserprop("wg_height",   2, wg_height   * 1e-6);
adduserprop("wg_width",    2, wg_width    * 1e-6);
adduserprop("ring_radius", 2, ring_radius * 1e-6);
adduserprop("gap",         2, gap         * 1e-6);
adduserprop("L_bus",       2, L_bus       * 1e-6);
adduserprop("add_drop",    0, add_drop);
adduserprop("material_core", 1, material_core);

ring_script = "

    # ---- Шина (through bus) вдоль x, центр по y = 0 ----
    addrect;
    set('name','bus_through');
    set('x', 0);
    set('x span', L_bus);
    set('y', 0);
    set('y span', wg_width);
    set('z', 0);
    set('z span', wg_height);
    set('material', material_core);

    # ---- Кольцо ----
    # Зазор gap отсчитывается край-в-край между шиной и кольцом:
    #   y_ring = gap + wg_width + ring_radius
    y_ring = gap + wg_width + ring_radius;

    addring;
    set('name','ring');
    set('x', 0);
    set('y', y_ring);
    set('z', 0);
    set('z span', wg_height);
    set('inner radius', ring_radius - wg_width/2);
    set('outer radius', ring_radius + wg_width/2);
    set('theta start', 0);
    set('theta stop', 360);
    set('material', material_core);

    # ---- Вторая шина (drop bus) для add-drop, зеркально сверху ----
    if (add_drop == 1) {
        addrect;
        set('name','bus_drop');
        set('x', 0);
        set('x span', L_bus);
        set('y', 2 * y_ring);
        set('y span', wg_width);
        set('z', 0);
        set('z span', wg_height);
        set('material', material_core);
    }

";

set("script", ring_script);

'''



def draw_ring_geometry() -> str:
    """lsf-скрипт структурной группы 'ring coupler'."""
    return _RING_GEOMETRY_LSF

