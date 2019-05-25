#!/usr/bin/python3

import os
from functools import reduce
from math import gcd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


def main(x1, x2, points, f):
    X = [x1, x2]
    document = open('lab_1.md', 'w')
    document.write(
        '---\n' +
        'mainfont: Google Sans\n' +
        'header-includes:\n' +
        r'- \usepackage{booktabs}'+'\n' +
        'output:\n' +
        'pdf_document:\n' +
        'toc: false\n' +
        'template: default.latex\n' +
        '---\n\n'
    )

    document.write(
        '# Лабораторная работа № 1\n' +
        '## Исходные данные\n\n' +
        '; '.join([
            '%s(%d, %d)' % (
                ['A', 'B', 'C', 'D', 'E'][i],
                p[x1],
                p[x2]
            ) for (i, p) in enumerate(points)
        ]) + '\n'
    )

    fig: plt.Figure = plt.figure(1, (4, 4), 100)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    colors = ['b', 'g', 'r', 'c']
    for i in range(1, len(points)):
        ax.plot(
            [points[i - 1][x1], points[i][x1]],
            [points[i - 1][x2], points[i][x2]],
            c=colors[i - 1]
        )
    ax.set_xlim(0, 100)
    ax.set_xlabel('$x_1$')
    ax.set_xticks(np.linspace(0, 100, 11, True))
    ax.set_ylim(0, 100)
    ax.set_ylabel('$x_2$')
    ax.set_yticks(np.linspace(0, 100, 11, True))
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.grid(True)
    if os.path.isfile('plot_1.png'):
        os.remove('plot_1.png')
    fig.savefig('plot_1.png', transparent=True)
    document.write('![Исходный график](plot_1.png)\n\n')

    # Целевая функция
    document.write(
        '## Целевая функция\n\n' +
        '$$ f = ' + sp.latex(f) + ' $$\n\n'
    )

    # Область ограничений
    systemX = []
    document.write(
        '## Область ограничений\n\n' +
        '### Стандартная форма\n' +
        r'$$ X = \begin{cases}'
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
        r' \geq 0. \end{cases} $$' + '\n\n'
    )

    document.write(
        '### Каноническая форма\n\n' +
        r'$$ X = \begin{cases}'
    )
    equationX = systemX.copy()
    for i in range(len(equationX)):
        x = sp.symbols('x%d' % (i + 3))
        X.append(x)
        equationX[i] = sp.Eq(equationX[i].lhs + x, equationX[i].rhs)
        document.write(
            r' {0},\\'.format(sp.latex(equationX[i]))
        )
    document.write(
        ','.join([sp.latex(x) for x in X]) +
        r' \geq 0. \end{cases}$$' + '\n\n'
    )

    __coeff = f.as_coefficients_dict()
    __x = np.linspace(-10, 10, 21)
    __y = (-__coeff[x1] * __x) / __coeff[x2]
    ax.plot(
        __x,
        __y,
        c='k',
        ls='dashed'
    )
    ax.plot(
        __x + max([p[x1] for p in points]),
        __y + max([p[x2] for p in points]),
        c='k',
        ls='dashed'
    )

    ax.set_xlim(-10, 100)
    ax.set_xlabel('$x_1$')
    ax.set_xticks(np.linspace(-10, 100, 12, True))
    ax.set_ylim(-10, 100)
    ax.set_ylabel('$x_2$')
    ax.set_yticks(np.linspace(-10, 100, 12, True))
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.grid(True)
    if os.path.isfile('plot_2.png'):
        os.remove('plot_2.png')
    fig.savefig('plot_2.png', transparent=True)
    fig.clear()
    document.write(
        '## График\n\n' +
        '![График](plot_2.png)\n\n'
    )

    table = pd.DataFrame(columns=['x_1', 'x_2', 'f(x)'])
    for i, p in enumerate(points):
        table.loc[i] = [p[x1], p[x2], f.subs({x1: p[x1], x2: p[x2]})]

    max_f = table['f(x)'].astype(float).idxmax()
    optX = {x1: table.at[max_f, 'x_1'], x2: table.at[max_f, 'x_2']}

    document.write(
        '## Значение целевой функции в точках пересечения прямых\n\n' +
        '$$' + table.to_latex() + '$$\n\n' +
        '## Таблица базисных переменных\n\n'
    )

    table = pd.DataFrame(columns=X + ['ДБР'])
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            new_row = {X[i]: 0, X[j]: 0}
            __symbols = X[0:i] + X[i + 1:j] + X[j + 1:]
            solution = sp.linsolve(equationX, __symbols).subs(
                {X[i]: 0, X[j]: 0}).as_dummy()
            if not solution:
                for x in X:
                    new_row[x] = '-'
                new_row['ДБР'] = '-'
            else:
                for s in solution:
                    for k, v in enumerate(s):
                        new_row[__symbols[k]] = v
                    new_row['ДБР'] = '-' if np.any(np.array(s) < 0) else '+'
            table = table.append(new_row, ignore_index=True)

    table.index += 1
    document.write(
        table.to_latex().replace('llllllll', 'c|c|c|c|c|c|c|c')
    )
    document.close()
    return systemX, X, optX
