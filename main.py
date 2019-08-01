#!/usr/bin/python3

import os

import sympy as sp

import lab_1
import lab_2
import lab_3
import lab_4

x1, x2 = sp.symbols("x1 x2")
f: sp.Add = 7 * x1 + 4 * x2
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
    {x1: sp.nsimplify(82 + 1/7), x2: 80},
    {x1: 100, x2: sp.nsimplify(62 + 1/7)}
]
lab_4.main(points.copy(), X, f, optX, systemX, simplexTable, activeLine)

print('Finish!')
