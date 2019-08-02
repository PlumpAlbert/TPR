import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

x1, x2 = sp.symbols('x1 x2')


def plot_new_limit(plot_points, limit_points, filename):
    figure: plt.Figure = plt.figure(2, (6, 6), 100)
    axes: plt.Axes = figure.add_subplot(1, 1, 1)
    colors = ['b', 'g', 'r', 'c']
    for i in range(1, len(plot_points)):
        axes.plot(
            [plot_points[i - 1][x1], plot_points[i][x1]],
            [plot_points[i - 1][x2], plot_points[i][x2]],
            c=colors[i - 1]
        )
    axes.plot(
        [limit_points[0][x1], limit_points[1][x1]],
        [limit_points[0][x2], limit_points[1][x2]],
        c='#FF5533'
    )
    axes.set_xlim(0, 100)
    axes.set_xlabel('$x_1$')
    axes.set_xticks(np.linspace(0, 110, 12, True))
    axes.set_ylim(0, 100)
    axes.set_ylabel('$x_2$')
    axes.set_yticks(np.linspace(0, 110, 12, True))
    axes.spines['top'].set_color('none')
    axes.xaxis.tick_bottom()
    axes.spines['right'].set_color('none')
    axes.yaxis.tick_left()
    axes.spines['left'].set_position('zero')
    axes.spines['bottom'].set_position('zero')
    axes.grid(True)
    if os.path.isfile(filename):
        os.remove(filename)
    figure.savefig(filename, transparent=True)
    axes.clear()
    figure.clear()


def main(
        points,
        X,
        f,
        optX,
        systemX,
        simplexTable,
        activeLine,
        excessLine
):
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
    new_line = r'\\'
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
{''.join([sp.latex(s) + f',{new_line}' for s in systemX]) +
 ','.join([sp.latex(x) for x in X[0:2]])}
\geq 0.
\end{{cases}}
$$

Найденное оптимальное решение в предыдущих лабораторных работах:
$$f^*=f(x^*)={f.subs(optX)}$$

Симплекс-таблица, полученная в лабораторной работе № 3:
$${pd.DataFrame(simplexTable, copy=True)
        .set_index('Базис')
        .reindex(X + ['f(x)'])
        .dropna().to_latex()}$$""" + '\n\n'
    )

    def introduce_new_limit(basic_simplex_table, limit_line, line_type):
        document.write("### Введем дополнительное ограничение\n")
        plot_new_limit(points, limit_line, f'lab4_{line_type}.png')
        document.write(f'![Новое ограничение](lab4_{line_type}.png)\n\n')
        lhs = (x1 - limit_line[0][x1]) / (limit_line[1][x1] - limit_line[0][x1])
        rhs = (x2 - limit_line[0][x2]) / (limit_line[1][x2] - limit_line[0][x2])
        coefficient = (lhs - rhs).as_coefficients_dict()
        limit_line_expression = sp.Eq(coefficient[x1] * x1 + coefficient[x2] * x2, -coefficient[1])
        document.write(f"""
Уравнение прямой: ${sp.latex(limit_line_expression)}$;

Ограничение: ${sp.latex(sp.LessThan(limit_line_expression.lhs, limit_line_expression.rhs))}$;

Представим в канонической форме: ${sp.latex(
            sp.Eq(limit_line_expression.lhs + sp.symbols("x7"), limit_line_expression.rhs))}$;

Так как $x_1$ и $x_2$ - базисные переменные, их необходимо исключить из уравнения."""
                       )
        x_1 = basic_simplex_table.loc[x1][0] - np.sum([
            c * X[i]
            if X[i] != x1
            else 0
            for i, c in enumerate(basic_simplex_table.loc[x1][1:])
        ])
        document.write(f"""
Выразим $x_1$ из {basic_simplex_table.index.to_list().index(x1) + 1}-й строки:
$$ x_1 = {sp.latex(x_1)} $$
        """)
        x_2 = basic_simplex_table.loc[x2][0] - np.sum([
            c * X[i]
            if X[i] != x2
            else 0
            for i, c in enumerate(basic_simplex_table.loc[x2][1:])
        ])
        document.write(f"""
Выразим $x_2$ из {basic_simplex_table.index.to_list().index(x2) + 1}-й строки:
$$ x_2 = {sp.latex(x_2)} $$
    """)
        limit_line_expression = sp.nsimplify(limit_line_expression.subs({x1: x_1, x2: x_2}))
        coefficient = limit_line_expression.lhs.as_coefficients_dict()
        x7 = sp.symbols('x7')
        if x7 not in X:
            X.append(x7)
        limit_line_expression = sp.Eq(limit_line_expression.lhs - coefficient[1] + X[6],
                                      limit_line_expression.rhs - coefficient[1])
        document.write(
            "$$" + sp.latex(limit_line_expression) + "$$" +
            "\nПолученное уравнение вводим в симплекс-таблицу"
        )
        basic_simplex_table[X[6]] = [0] * len(basic_simplex_table)
        basic_simplex_table.loc[X[6]] = [limit_line_expression.rhs] + [
            limit_line_expression.lhs.as_coefficients_dict()[x] for x in X]
        basic_simplex_table = basic_simplex_table.reindex(X + ['f(x)']).dropna().reset_index()

        def print_table(table):
            t = pd.DataFrame(table, copy=True)
            document.write(
                t.set_index('Базис').to_latex()
            )

        def recalc(simplex_table):
            while not np.all(simplex_table.iloc[-1][1:] >= 0) \
                    or not np.all(simplex_table['B'] >= 0):
                # Находим ведущий столбец
                columns = simplex_table.iloc[-1][1:].astype(float)
                columns[columns == 0] = np.Infinity
                leadCol = columns.idxmin()
                # print(table[leadCol])
                # Находим ведущую строку
                leadRow = np.argmin([
                    simplex_table.at[i, 'B'] / row
                    if row != 0
                    else np.Infinity
                    for i, row in enumerate(simplex_table[leadCol][:-1])
                ])
                # Находим ведущий элемент
                leadElem = simplex_table.at[leadRow, leadCol]

                document.write(
                    'Ведущий столбец: $' + sp.latex(leadCol) + '$\n' +
                    'Ведущая строка: $' + sp.latex(simplex_table.at[leadRow, 'Базис']) + '$\n' +
                    'Ведущий элемент: $' + sp.latex(leadElem) + '$\n\n'
                )

                # Пересчитываем элементы таблицы
                # Делим элементы главной строки на ведущий элемент
                simplex_table.at[leadRow, 'Базис'] = leadCol
                for col in simplex_table.columns[1:]:
                    simplex_table.at[leadRow, col] /= leadElem

                for row in range(len(simplex_table) - 1):
                    if row == leadRow:
                        continue
                    aik = simplex_table.at[row, leadCol]
                    for col in simplex_table.columns[1:]:
                        simplex_table.at[row, col] -= simplex_table.at[leadRow, col] * aik
                        document.write(
                            'Cell[{0}][{1}] -= {2} * {3} = {4}\n\n'.format(
                                simplex_table.at[row, 'Базис'],
                                col,
                                simplex_table.at[leadRow, col],
                                aik,
                                simplex_table.at[row, col]
                            )
                        )

                aik = simplex_table.at[len(simplex_table) - 1, leadCol]
                simplex_table.at[len(simplex_table) - 1, 'B'] -= simplex_table.at[leadRow, 'B'] * aik
                for col in simplex_table.columns[2:]:
                    simplex_table.at[len(simplex_table) - 1, col] -= simplex_table.at[leadRow, col] * aik

                print_table(simplex_table)

        print_table(basic_simplex_table)
        recalc(pd.DataFrame(basic_simplex_table, copy=True))

    introduce_new_limit(
        pd.DataFrame(simplexTable, copy=True)
            .set_index('Базис')
            .reindex(X + ['f(x)'])
            .dropna(),
        activeLine,
        'active'
    )
    document.write("""\n\n
В симплекс-таблице в столбце базисных переменных нет отрицательных элементов,
значит данное базисное решение оптимально. 

Значение целевой функции ухудшилось по сравнению с исходным оптимальным решением,
значит дополнительное ограничение ***активное***.\n\n""")

    introduce_new_limit(
        pd.DataFrame(simplexTable, copy=True)
            .set_index('Базис')
            .reindex(X + ['f(x)'])
            .dropna(),
        excessLine,
        'excess'
    )
    document.write("""\n\n
В симплекс-таблице в столбце базисных переменных нет отрицательных элементов,
значит данное базисное решение оптимально. 

Значение целевой функции не изменилось по сравнению с исходным оптимальным решением.
С помощью графического способа можно убедиться, что оптимальное решение лежит на прямой,
являющейся дополнительным ограничением. Следовательно, дополнительное ограничение ***избыточное***.\n\n""")

    introduce_new_limit(
        pd.DataFrame(simplexTable, copy=True)
            .set_index('Базис')
            .reindex(X + ['f(x)'])
            .dropna(),
        excessLine,
        'excess'
    )
    document.write("""\n\n
В симплекс-таблице в столбце базисных переменных нет отрицательных элементов,
значит данное базисное решение оптимально. 

Значение целевой функции не изменилось по сравнению с исходным оптимальным решением.
С помощью графического способа можно убедиться, что оптимальное решение лежит на прямой,
являющейся дополнительным ограничением. Следовательно, дополнительное ограничение ***избыточное***.\n\n""")
