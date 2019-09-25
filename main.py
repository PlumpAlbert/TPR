#!/usr/bin/python3

import sympy as sp

import lab_1
import lab_2
import lab_3
import lab_4

x1, x2 = sp.symbols("x1 x2")
f: sp.Add = 5 * x1 + 6 * x2
# Исходные точки
points = [
    {x1: 0, x2: 80},
    {x1: 100, x2: 80},
    {x1: 100, x2: 50},
    {x1: 80, x2: 30},
    {x1: 30, x2: 0}
]

# Лабораторная работа № 1

systemX, X, optX = lab_1.main(x1, x2, points, f)

# Конец ЛР 1

# Лабораторная работа № 2

lab_2.main(f, systemX, X, optX)

# Конец ЛР 2

simplexTable = lab_3.main(X, systemX, f, optX)

activeLine = [
    {x1: sp.nsimplify(76 + 7 / 8), x2: 80},
    {x1: 100, x2: sp.nsimplify(62 + 3 / 4)}
]
excess_start_point = {
    # x1: optX[x1] - sp.nsimplify(uniform(0, 20)),
    # x2: optX[x2] + sp.nsimplify(uniform(0, 20))
    x1: 90, x2: 90
}
excess_line_y = ((x1 - excess_start_point[x1])
                 * (optX[x2] - excess_start_point[x2])
                 / (optX[x1] - excess_start_point[x1])
                 ) + excess_start_point[x2]
excessLine = [
    excess_start_point,
    {x1: 110, x2: excess_line_y.subs({x1: 110})}
]
inactiveLine = [
    {
        x1: 32 + sp.nsimplify(5 / 6),
        x2: 20
    },
    {
        x1: 90,
        x2: 25 + sp.nsimplify(3 / 7)
    }
]
lab_4.main(
    points.copy(),
    X,
    f,
    optX,
    systemX,
    simplexTable,
    activeLine,
    excessLine,
    inactiveLine
)

print('Finish!')
