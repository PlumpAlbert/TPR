import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


def main(points, X, f, optX, systemX, simplexTable, activeLine):
    document = open('lab_4.md', 'w')
    document.write(
        '---\n' +
        'mainfont: CMU Typewriter Text\n' +
        'header-includes:\n' +
        r'- \usepackage{booktabs}' + '\n' +
        'output:\n' +
        'pdf_document:\n' +
        'toc: false\n' +
        'template: default.latex\n' +
        '---\n\n'
    )
    simplexTable = pd.DataFrame(simplexTable, copy=True) \
        .set_index('Базис') \
        .reindex(X + ['f(x)']) \
        .dropna()
    newLine = r'\\'
    document.write(
        rf"""# Лабораторная работа № 4

## Цель работы

Проанализировать имеющуюся линейную модель на чувствительность, используя 
двойственный симплекс метод.

## Решение

![Область ограничений](plot_1.png)

Целевая функция:
$$f(x)={sp.latex(f)}$$

Область ограничений задачи в стандартной форме:
$$
X=\begin{{cases}}
{''.join([sp.latex(s) + f',{newLine}' for s in systemX]) +
 ','.join([sp.latex(x) for x in X[0:2]])}
\geq 0.
\end{{cases}}
$$

Найденное оптимальное решение в предыдущих лабораторных работах:
$$f^*=f(x^*)={f.subs(optX)}$$

Симплекс-таблица, полученная в лабораторной работе № 3:
$${simplexTable.to_latex()}$$""" + '\n\n'
    )
    x1, x2 = X[0:2]
    document.write("### Введем дополнительное ограничение\n")
    figure: plt.Figure = plt.figure(2, (4, 4), 100)
    axes: plt.Axes = figure.add_subplot(1, 1, 1)
    colors = ['b', 'g', 'r', 'c']
    for i in range(1, len(points)):
        axes.plot(
            [points[i - 1][x1], points[i][x1]],
            [points[i - 1][x2], points[i][x2]],
            c=colors[i - 1]
        )
    axes.plot(
        [activeLine[0][x1].evalf(), activeLine[1][x1]],
        [activeLine[0][x2], activeLine[1][x2].evalf()],
        c='#FF5733'
    )
    axes.set_xlim(0, 100)
    axes.set_xlabel('$x_1$')
    axes.set_xticks(np.linspace(0, 100, 11, True))
    axes.set_ylim(0, 100)
    axes.set_ylabel('$x_2$')
    axes.set_yticks(np.linspace(0, 100, 11, True))
    axes.spines['top'].set_color('none')
    axes.xaxis.tick_bottom()
    axes.spines['right'].set_color('none')
    axes.yaxis.tick_left()
    axes.spines['left'].set_position('zero')
    axes.spines['bottom'].set_position('zero')
    axes.grid(True)
    if os.path.isfile('lab4_1.png'):
        os.remove('lab4_1.png')
    figure.savefig('lab4_1.png', transparent=True)
    document.write('![Новое ограничение](lab4_1.png)\n\n')
    lhs = (x1 - activeLine[0][x1]) / (activeLine[1][x1] - activeLine[0][x1])
    rhs = (x2 - activeLine[0][x2]) / (activeLine[1][x2] - activeLine[0][x2])
    coeff = (lhs - rhs).as_coefficients_dict()
    newLine = sp.Eq(coeff[x1] * x1 + coeff[x2] * x2, -coeff[1])
    document.write(f"""
Уравнение прямой: ${sp.latex(newLine)}$;

Ограничение: ${sp.latex(sp.LessThan(newLine.lhs, newLine.rhs))}$;

Представим в канонической форме: ${sp.latex(sp.Eq(newLine.lhs + sp.symbols("x7"), newLine.rhs))}$;

Так как $x_1$ и $x_2$ - базисные переменные, их необходимо исключить из уравнения.
""")
    x_1 = simplexTable.loc[x1][0] - np.sum([
        c * X[i]
        if X[i] != x1
        else 0
        for i, c in enumerate(simplexTable.loc[x1][1:])
    ])
    document.write(f"""
Выразим $x_1$ из {simplexTable.index.to_list().index(x1) + 1}-й строки:
$$ x_1 = {sp.latex(x_1)} $$
    """)
    x_2 = simplexTable.loc[x2][0] - np.sum([
        c * X[i]
        if X[i] != x2
        else 0
        for i, c in enumerate(simplexTable.loc[x2][1:])
    ])
    document.write(f"""
Выразим $x_2$ из {simplexTable.index.to_list().index(x2) + 1}-й строки:
$$ x_2 = {sp.latex(x_2)} $$
""")
    newLine = newLine.subs({x1: x_1, x2: x_2})
    coeff = newLine.lhs.as_coefficients_dict()
    X.append(sp.symbols('x7'))
    newLine = sp.Eq(newLine.lhs - coeff[1] + X[6], newLine.rhs - coeff[1])
    document.write(
        "$$" + sp.latex(newLine) + "$$" +
        "\nПолученное уравнение вводим в симплекс-таблицу$$"
    )
    simplexTable[X[6]] = [0] * len(simplexTable)
    o = pd.Series(simplexTable.iloc[len(simplexTable) - 1], copy=True)
    simplexTable.loc[X[6]] = [newLine.rhs] + [newLine.lhs.as_coefficients_dict()[x] for x in X]
    simplexTable = simplexTable.reindex(X + ['f(x)']).dropna().reset_index()

    def printTable(table):
        t = pd.DataFrame(table, copy=True)
        document.write(
            t.set_index('Базис').to_latex()
        )

    printTable(simplexTable)
    document.write("$$")
    while not np.all(simplexTable.iloc[len(simplexTable) - 1][1:] >= 0) \
            or not np.all(simplexTable['B'] >= 0):
        # Находим ведущий столбец
        columns = simplexTable.iloc[-1][1:].astype(float)
        columns[columns == 0] = np.Infinity
        leadCol = columns.idxmin()
        # print(table[leadCol])
        # Находим ведущую строку
        leadRow = np.argmin([
            simplexTable.at[i, 'B'] / row
            if row != 0
            else np.Infinity
            for i, row in enumerate(simplexTable[leadCol][:-1])
        ])
        # Находим ведущий элемент
        leadElem = simplexTable.at[leadRow, leadCol]

        document.write(
            'Ведущий столбец: $' + sp.latex(leadCol) + '$\n' +
            'Ведущая строка: $' + sp.latex(simplexTable.at[leadRow, 'Базис']) + '$\n' +
            'Ведущий элемент: $' + sp.latex(leadElem) + '$\n\n'
        )

        # Пересчитываем элементы таблицы
        # Делим элементы главной строки на ведущий элемент
        simplexTable.at[leadRow, 'Базис'] = leadCol
        for col in simplexTable.columns[1:]:
            simplexTable.at[leadRow, col] /= leadElem

        for row in range(len(simplexTable) - 1):
            if row == leadRow:
                continue
            aik = simplexTable.at[row, leadCol]
            for col in simplexTable.columns[1:]:
                simplexTable.at[row, col] -= simplexTable.at[leadRow, col] * aik
                document.write(
                    'Cell[{0}][{1}] -= {2} * {3} = {4}\n\n'.format(
                        simplexTable.at[row, 'Базис'],
                        col,
                        simplexTable.at[leadRow, col],
                        aik,
                        simplexTable.at[row, col]
                    )
                )

        aik = simplexTable.at[len(simplexTable) - 1, leadCol]
        simplexTable.at[len(simplexTable) - 1, 'B'] -= simplexTable.at[leadRow, 'B'] * aik
        for col in simplexTable.columns[2:]:
            simplexTable.at[len(simplexTable) - 1, col] -= simplexTable.at[leadRow, col] * aik

        printTable(simplexTable)
    document.write("""\n\n
В симплекс-таблице в столбце базисных переменных нет отрицательных элементов,
значит данное базисное решение оптимально. 

Значение целевой функции ухудшилось по сравнению с исходным оптимальным решением,
значит дополнительное ограничение ***активное***.""")
