import pandas as pd
import sympy as sp


def main(f, systemX, X, optX):
    x1, x2 = X[0:2]
    document = open('lab_2.md', 'w')
    
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
        '# Осуществим переход к двойственной задаче\n' +
        'Целевая функция: $f(x)=' + sp.latex(f) + '$\n\n' +
        'Область ограничений в стандартной форме:\n' +
        r'$$X = \begin{cases}' +
        r',\\'.join([sp.latex(s) for s in systemX]) + r',\\' +
        ','.join([sp.latex(x) for x in X]) +
        r'\geq 0.\end{cases}\\$$'
    )

    Y = []
    phi = 0

    for i, coeff in enumerate([s.rhs for s in systemX]):
        Y.append(sp.symbols('y%d' % (i + 1)))
        phi += Y[i] * coeff

    systemY = []
    for x in (x1, x2):
        systemY.append(0)
        for i, s in enumerate(systemX):
            systemY[-1] += s.lhs.coeff(x) * Y[i]
        systemY[-1] = sp.GreaterThan(systemY[-1], f.coeff(x))

    document.write(
        'Двойственная задача:\n' +
        r'$$\varphi(' +
        ','.join([sp.latex(y) for y in Y])
        + ')=' + sp.latex(phi) + r'\rightarrow min $$' +
        r'$$Y=\begin{cases}' +
        r',\\'.join([sp.latex(s) for s in systemY]) + r',\\' +
        ','.join([sp.latex(y) for y in Y]) + r'\geq 0.'
        r'\end{cases}$$'
    )

    document.write(
        '\n\n# Используя теоремы двойственности, \
      найдем решение задачи линейного программирования\n' +
        'Найденное оптимальное решение в лабораторной работе № 1 :\n\n' +
        '$x^* = (%d, %d)$' % (optX[x1], optX[x2])
    )
