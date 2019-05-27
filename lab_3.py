import numpy as np
import pandas as pd
import sympy as sp


def main(X, systemX, f, optX):
    document = open('lab_3.md', 'w')
    document.write(
        '---\n' +
        'mainfont: Google Sans\n' +
        'header-includes:\n' +
        r'- \usepackage{booktabs}' + '\n' +
        'output:\n' +
        'pdf_document:\n' +
        'toc: false\n' +
        'template: default.latex\n' +
        '---\n\n'
    )

    document.write(
        '# Исходные данные\n\n' +
        'Область ограничений задачи в стандартной форме:\n' +
        r'$$ X = \begin{cases}' +
        r',\\'.join([sp.latex(e) for e in systemX]) +
        r',\\ x_1, x_2 \geq 0.\end{cases}$$' + '\n\n'
    )

    document.write('Приведем задачу к канонической форме\n\n')

    f = f + np.sum([0 * p for p in X[2:]])
    document.write(
        '$$f(x)=' +
        sp.latex(f) + '+ 0' +
        '+0'.join([sp.latex(p) for p in X[2:]]) +
        r'\rightarrow max$$' + '\n\n'
    )

    document.write(
        r'$$X=\begin{cases}' +
        r',\\'.join([
            sp.latex(sp.Eq(
                np.sum([
                    sp.sympify(s.lhs.coeff(p)) * p
                    if j != i + 2
                    else p
                    for j, p in enumerate(X)
                ]),
                s.rhs
            ))
            for i, s in enumerate(systemX)
        ]) +
        r'.\end{cases}$$' + '\n\n'
    )

    for i, s in enumerate(systemX):
        systemX[i] = sp.Eq(
            np.sum([
                s.lhs.coeff(p) * p
                if j != i + 2
                else p
                for j, p in enumerate(X)
            ]),
            s.rhs
        )
    document.write(
        'Базисные переменные: $' +
        ','.join([sp.latex(e) for e in X[2:]]) + '$\n\n' +
        'Небазисные переменные: $' +
        ','.join([sp.latex(e) for e in X[:2]]) + '$\n\n'
    )

    document.write('# Исходная симплекс-таблица\n\n')
    # Create table
    table = pd.DataFrame(columns=['Базис', 'B'] + [e for e in X])

    # Init table with default values
    for i in range(len(X[2:])):
        table.at[i, 'Базис'] = X[i + 2]
        table.at[i, 'B'] = systemX[i].rhs
        for p in X:
            table.at[i, p] = systemX[i].lhs.coeff(p)

    table.at[i + 1, 'Базис'] = 'f(x)'
    table.at[i + 1, 'B'] = 0
    for p in X:
        table.at[i + 1, p] = -f.coeff(p)

    def printTable(table):
        t = pd.DataFrame(table, copy=True)
        document.write(
            t.set_index('Базис').to_latex()
        )

    printTable(table)

    while not np.all(table.iloc[len(table) - 1][1:] >= 0):
        # Находим ведущий столбец
        leadCol = table.iloc[-1][1:].astype(float).idxmin()
        # print(table[leadCol])
        # Находим ведущую строку
        leadRow = np.argmin([
            table.at[i, 'B'] / row
            if row > 0
            else np.Infinity
            for i, row in enumerate(table[leadCol])
        ])
        # Находим ведущий элемент
        leadElem = table.at[leadRow, leadCol]

        document.write(
            'Ведущий столбец: $' + sp.latex(leadCol) + '$\n' +
            'Ведущая строка: $' + sp.latex(table.at[leadRow, 'Базис']) + '$\n' +
            'Ведущий элемент: $' + sp.latex(leadElem) + '$\n\n'
        )

        # Пересчитываем элементы таблицы
        # Делим элементы главной строки на ведущий элемент
        table.at[leadRow, 'Базис'] = leadCol
        for col in table.columns[1:]:
            table.at[leadRow, col] /= leadElem

        for row in range(len(table) - 1):
            if row == leadRow:
                continue
            aik = table.at[row, leadCol]
            for col in table.columns[1:]:
                table.at[row, col] -= table.at[leadRow, col] * aik
                document.write(
                    'Cell[{0}][{1}] -= {2} * {3} = {4}\n\n'.format(
                        table.at[row, 'Базис'],
                        col,
                        table.at[leadRow, col],
                        aik,
                        table.at[row, col]
                    )
                )

        aik = table.at[len(table) - 1, leadCol]
        table.at[len(table) - 1, 'B'] -= table.at[leadRow, 'B'] * aik
        for col in table.columns[2:]:
            table.at[len(table) - 1, col] -= table.at[leadRow, col] * aik

        printTable(table)
