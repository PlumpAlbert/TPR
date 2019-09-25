import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import rc

x1, x2 = sp.symbols('x1 x2')
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
document = open('lab_4.tex', 'w')
document.write(
    r'\documentclass[12pt,a4paper]{article}' '\n'
    r'\renewcommand{\figurename}{Рис.}''\n'
    r"\usepackage{amsmath}" "\n"
    r'\usepackage{booktabs}' '\n'
    r'\usepackage{graphicx}' '\n'
    r'\usepackage{xecyr}''\n'
    r"\defaultfontfeatures{Mapping=tex-text,Scale=MatchLowercase}" "\n"
    r"\setmainfont{Monofur Nerd Font Mono}" "\n"

    r'\title{Лабораторная работа №4}' '\n'
    r'\author{Plump Albert}' '\n'
    r'\date{September 2019}' '\n'

    r'\begin{document}' '\n'
    r'\begin{titlepage}' '\n'
    r'\maketitle' '\n'
    r'\newpage' '\n'
    r'\end{titlepage}''\n'
)


def simplex_shit(table, X, double=False, artificial=False):
    iteration = 0
    if artificial:
        omega_index = table.index[table['Базис'] == '-w(x)'][0]
    while np.any(table.iloc[-1][1:] < 0) or double:
        if artificial and table.iloc[omega_index]['B'] == 0:
            artificial = False
            table = table.drop(omega_index, axis=0) \
                .drop(table.index[table['Базис'] == X[7]], axis=0) \
                .drop(X[7], axis=1)
            table.index = range(0, len(table))
            X.pop()

        if double:
            document.write(
                r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n' +
                r'\begin{center}' +
                pd.DataFrame(table, copy=True)
                .set_index('Базис')
                .to_latex() +
                r'\end{center}'
                r'\mbox{}\\В симплекс-таблице в столбце базисных переменных есть отрицательные элементы, '
                r'значит используем алгоритм двойственного симплекс-метода.'
            )
            leadRow = table.iloc[:-1]['B'].astype(float).idxmin()
            leadCol = table.columns[np.argmin([
                table.at[len(table) - 1, _x] / table.at[leadRow, _x]
                if table.at[leadRow, _x] != 0 and table.at[len(table) - 1, _x] != 0
                else np.Infinity
                for _x in X
            ]) + 2]
        else:
            document.write(
                r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n' +
                r'\begin{center}' +
                pd.DataFrame(table, copy=True)
                .set_index('Базис')
                .to_latex() +
                r'\end{center}'
                r'\mbox{}\\В симплекс-таблице есть отрицательные коэффициенты строки $f(x)$, '
                r'значит данное базисное решение не оптимально.'
            )
            # Находим ведущий столбец
            leadCol = table.iloc[-1][1:].astype(float).idxmin()
            # print(table[leadCol])
            # Находим ведущую строку
            leadRow = np.argmin([
                table.at[i, 'B'] / row
                if row > 0
                else np.Infinity
                for i, row in enumerate(table[leadCol][:-1])
            ])
        iteration += 1

        # Находим ведущий элемент
        leadElem = table.at[leadRow, leadCol]
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

        aik = table.at[len(table) - 1, leadCol]
        table.at[len(table) - 1, 'B'] -= table.at[leadRow, 'B'] * aik
        for col in table.columns[2:]:
            table.at[len(table) - 1, col] -= table.at[leadRow, col] * aik
        if np.all(table['B'] >= 0):
            double = False

    _t = pd.DataFrame(table, copy=True).set_index('Базис')
    document.write(
        r'\subparagraph{Итерация ' + str(iteration + 1) + r'}\mbox{}\\''\n' +
        r'\begin{center}' +
        _t.to_latex() +
        r'\end{center}'
        r'\mbox{}\\В симплекс-таблице все коэффициенты строки $f(x)$ неотрицательные, '
        r'значит данное базисное решение оптимально.\\'
        rf'Таким образом, $f^* = f({_t.at[x1, "B"]}, {_t.at[x2, "B"]}) = {_t.at["f(x)", "B"]}$.'
    )


def main(
        points,
        X,
        f,
        optX,
        systemX,
        simplexTable,
        activeLine,
        excessLine,
        inactiveLine
):
    new_line = r'\\'

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(
        [el[x1] for el in points],
        [el[x2] for el in points],
        c='DODGERBLUE',
        lw=2,
        marker='o',
        mfc='DODGERBLUE',
        mec='DODGERBLUE'
    )
    x_line = np.linspace(-20, 20, 3)
    y_line = - 7 / 4 * x_line
    plt.plot(
        x_line,
        y_line,
        c='CORAL',
        lw=2
    )
    plt.xlim(-20, 105)
    plt.ylim(-20, 105)
    plt.grid(True)
    plt.axhline(y=0, color='k', lw=2)
    plt.axvline(x=0, color='k', lw=2)
    plt.tight_layout()
    plt.savefig('plot_l4.png')
    document.write(
        r'\section{Задание}'
        r'\begin{enumerate}'
        r'\item В исходную ЗЛП добавить новое ограничение таким образом, чтобы базисные переменные в соответствии с '
        'ограничениями на начальной итерации получили отрицательные значения.'
        r'\item Найти решение поставленной ЗЛП 3-мя способами:'
        r'\begin{itemize}'
        r'\item Используя обычный симплекс-метод;'
        r'\item Используя двойственный симплекс-метод;'
        r'\item Ввести дополнительную неотрицательную искусственную переменную и найти решение с помощью '
        r'симплекс-метода, пересчитывая на каждой итерации значение $f(x)$.'
        r'\end{itemize}'
        r'\end{enumerate}'
    )
    document.write(
        r'\section{Решение}'
        # r'\clearpage''\n'
        r'\begin{figure}[h!]'
        r'\centering'
        r'\includegraphics[height=10cm]{plot_l4.png}'
        r'\caption{Область ограничений}'
        r'\end{figure}'
    )
    document.write(
        r'Целевая функция имеет вид: $f(x_1,x_2)='
        + sp.latex(f) +
        r'\rightarrow max.$\\'
        r'А область ограничений задачи в стандартной форме имеет вид:'
        r'\begin{equation*}''\n'
        r'X = \begin{cases}''\n' +
        ''.join([sp.latex(s).replace('=', '&=') + f',{new_line}' for s in systemX]) +
        ','.join([sp.latex(x) for x in X[0:2]])
        + r' &\geq 0.'
          r'\end{cases}''\n'
          r'\end{equation*}''\n'
    )
    document.write(
        r'Найденное оптимальное рещение в предыдущих практических работах:'
        r'$$ f^* = f_{max} = f(100, 80) = 1020 $$'
    )
    document.write(
        r'\subsection{Введем дополнительное ограничение}''\n'
    )
    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(
        [el[x1] for el in points],
        [el[x2] for el in points],
        c='DODGERBLUE',
        lw=2,
        marker='o',
        mfc='DODGERBLUE',
        mec='DODGERBLUE'
    )
    y_line = np.linspace(30, 90, 2)
    x_line = -9 * (y_line - 40) / 4 + 100
    plt.plot(
        x_line,
        y_line,
        c='CORAL',
        lw=2
    )
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.grid(True)
    plt.axhline(y=0, color='k', lw=2)
    plt.axvline(x=0, color='k', lw=2)
    plt.tight_layout()
    plt.savefig('plot_l4_2.png')
    document.write(
        r'\begin{figure}[h!]'
        r'\centering'
        r'\includegraphics[height=10cm]{plot_l4_2.png}'
        r'\caption{Новое ограничение}'
        r'\end{figure}'
        r'Уравнение прямой имеет вид: '
        r'$$4 x_1 + 9 x_2 = 760$$'
        r'Ограничение имеет вид: '
        r'$$4 x_1 + 9 x_2 \leq 760$$'
    )
    X.append(sp.symbols('x7'))
    systemX.append(
        sp.Eq(
            4 * x1 + 9 * x2 + X[6],
            760
        )
    )
    document.write(
        r'\begin{equation*}''\n'
        r'X = \begin{cases}''\n' +
        ''.join([sp.latex(s).replace('=', '&=') + f',{new_line}' for s in systemX]) +
        ','.join([sp.latex(x) for x in X])
        + r' &\geq 0.'
          r'\end{cases}''\n'
          r'\end{equation*}''\n'
          r'Среди переменных задачи можно выделить базисные переменные: $' +
        ','.join([sp.latex(s) for s in X[2:]]) +
        r'$ и не базисные: $' +
        ','.join([sp.latex(s) for s in X[0:2]]) + '$'
    )

    document.write(
        r'\paragraph{Используем обычный симплекс-метод для решения задачи}\mbox{}\\'
        r'$ a^{`}_{ij} = a_{ij} - \frac{a_{rj} * a_{ik}}{a_{rk}} $\\'
        r'$ b^{`}_{i} = b_{i} - \frac{b_{r} * a_{ik}}{a_{rk}} $\\'
        r'$ f^{`}(x) = f(x) - \frac{b_{r} * \Delta_{k}}{a_{rk}} $\\'
        r'$ \Delta^{`}_{j} = \Delta_{j} - \frac{a_{rj} * \Delta_{k}}{a_{rk}} $\\'
    )
    table = pd.DataFrame(simplexTable, copy=True)
    series = pd.Series(table.iloc[4], copy=True)
    table.iloc[4] = [X[6], 760, 4, 9, 0, 0, 0, 0]
    table = table.append(series, ignore_index=True)
    table[X[6]] = [0, 0, 0, 0, 1, 0]
    simplex_shit(pd.DataFrame(table, copy=True), X)

    document.write(
        r'\paragraph{Используем двойственный симплекс-метод для решения задачи}'
        r'\mbox{}\\'
    )
    simplex_shit(pd.DataFrame(table, copy=True), X, double=True)

    document.write(
        r'\paragraph{Используем искусственную переменную для решения задачи}'
        r'\mbox{}\\'
        r'Введем в левую часть ограничения $4 x_1 + 9 x_2 \leq 760$ неотрицательную '
        r'искусственную переменную $x_8$:'
    )
    X.append(sp.symbols('x8'))
    systemX[-1] = sp.Eq(systemX[-1].lhs + X[7], systemX[-1].rhs)
    omega = systemX[-1].rhs - systemX[-1].lhs + X[7]
    document.write(
        r'\begin{equation*}''\n'
        r'X = \begin{cases}''\n' +
        ''.join([sp.latex(s).replace('=', '&=') + f',{new_line}' for s in systemX]) +
        ','.join([sp.latex(x) for x in X])
        + r' &\geq 0.'
          r'\end{cases}''\n'
          r'\end{equation*}''\n'
          r'Для обращения в ноль искусственной переменной $x_8$ минимизируем '
          r'симплекс-методом искусственную целевую функцию $\omega(x) = x_8$. '
          r'Используем соотношение: $min(\omega(x)) = max(-\omega(x))$\\'
          r'Выразим $\omega(x)$ через небазисные переменные:\\$'
          r'\omega(x) = x_8 = ' + sp.latex(omega) + r'$\\' +
        r'Тогда $-\omega(x)=-x_8=' + sp.latex(-omega) + r'$\\'
    )
    _t = pd.DataFrame(table, copy=True)
    series = pd.Series(_t.iloc[-1], copy=True)
    # Меняем строку с x7 на x8
    _t.at[len(_t) - 2, 'Базис'] = X[7]
    # Добавляем строку w(x)
    _t.iloc[-1] = ['-w(x)', -760, -4, -9, 0, 0, 0, 0, -1]
    _t = _t.append(series, ignore_index=True)
    # Добавляем столбец x8
    _t[X[7]] = [0, 0, 0, 0, 1, 0, 0]
    simplex_shit(_t, X, artificial=True)
    document.write(r'\end{document}')
    return
