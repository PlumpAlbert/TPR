#!/usr/bin/python3

import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import gcd
from functools import reduce

# Лабораторная работа № 1
document = open('lab_1.md', 'w')
x1, x2 = sp.symbols("x1 x2")
X = [x1, x2]
# Исходные точки
points = [
    {x1: 0,  x2: 80},
    {x1: 90, x2: 80},
    {x1: 80, x2: 60},
    {x1: 55, x2: 20},
    {x1: 25, x2: 0}
]

document.write(
    '# Лабораторная работа № 1\n' +
    '## Исходные точки\n' +
    ', '.join(['(%d, %d)' % (p[x1], p[x2]) for p in points]) + '\n'
)

# Целевая функция
f = 5 * x1 + 6 * x2
document.write(
    '## Целевая функция\n' +
    '$ f = ' + sp.latex(f) + '$\n'
)

# Область ограничений
systemX = []
document.write(
    '## Область ограничений\n' +
    '### Стандартная форма\n' +
    r'$X = \begin{cases}'
)
for i in range(len(points) - 1):
    lhs = (x1 - points[i][x1]) * (points[i + 1][x2] - points[i][x2])
    rhs = (x2 - points[i][x2]) * (points[i + 1][x1] - points[i][x1])
    expr = lhs - rhs
    coeff = expr.as_coefficients_dict()
    cd = -reduce(gcd, [coeff[x1], coeff[x2], coeff[1]])
    expr = sp.LessThan(coeff[x1] * x1 / cd + coeff[x2]
                       * x2 / cd, -coeff[1] / cd)
    systemX.append(expr)
    document.write(
        r' {0},\\'.format(sp.latex(expr))
    )
document.write(
    ','.join([sp.latex(x) for x in X]) +
    r' \geq 0. \end{cases}$' + '\n'
)

document.write(
    '### Каноническая форма\n' +
    r'$ X = \begin{cases}'
)
for i in range(len(systemX)):
    x = sp.symbols('x%d' % (i + 3))
    X.append(x)
    systemX[i] = sp.Eq(systemX[i].lhs + x, systemX[i].rhs)
    document.write(
        r' {0},\\'.format(sp.latex(systemX[i]))
    )
document.write(
    ','.join([sp.latex(x) for x in X]) +
    r' \geq 0. \end{cases}$' + '\n'
)
