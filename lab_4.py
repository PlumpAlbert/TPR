import os
import matplotlib.pyplot as plt
import numpy as np
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
    print("lhs:", lhs)
    print("rhs:", rhs)
    coeff = (lhs - rhs).as_coefficients_dict()
    newLine = sp.Eq(coeff[x1] * x1 + coeff[x2] * x2, -coeff[1])
    document.write(f"""
Уравнение прямой: ${sp.latex(newLine)}$;

Ограничение: ${sp.latex(sp.LessThan(newLine.lhs, newLine.rhs))}$;

Представим в канонической форме: ${sp.latex(sp.Eq(newLine.lhs + sp.symbols("x7"), newLine.rhs))}$;

Так как $x_1$ и $x_2$ - базисные переменные, их необходимо исключить из уравнения.
""")
