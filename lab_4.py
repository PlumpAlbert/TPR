import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import rc

x1, x2 = sp.symbols('x1 x2')
_f = 5 * x1 + 6 * x2
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
document = open('lab_4.tex', 'w')
document.write(
    r'\documentclass[12pt,a4paper]{article}' '\n'
    r'\renewcommand{\figurename}{Рис.}''\n'
    r"\usepackage{amsmath}" "\n"
    r'\usepackage{booktabs}' '\n'
    r'\usepackage{graphicx}' '\n'
    r'\usepackage[english,russian]{babel}''\n'
    r'\usepackage{fontspec}''\n'
    r'\usepackage{mathptmx}''\n'
    r'\defaultfontfeatures{Ligatures={TeX},Renderer=Basic}''\n'
    r'\setmainfont[Ligatures={TeX,Historic}]{Monofur Nerd Font}''\n'
    r'\usepackage{xecyr}''\n'
    r"\defaultfontfeatures{Mapping=tex-text,Scale=MatchLowercase}" "\n"
    r"\setmainfont{Times New Roman}" "\n"

    r'\title{Лабораторная работа №4}' '\n'
    r'\author{Plump Albert}' '\n'
    r'\date{September 2019}' '\n'

    r'\begin{document}' '\n'
    r'\begin{titlepage}' '\n'
    r'\maketitle' '\n'
    r'\newpage' '\n'
    r'\end{titlepage}''\n'
)


def print_table(table):
    document.write(
        r'\begin{table}[!ht]''\n'
        r'\begin{center}''\n'
        r'\begin{tabular}{|l|'
        f'{len(table.columns) * "|c"}|''}\n'
        r'\toprule''\n' +
        f'{table.index.name} & {" & ".join(["$" + sp.latex(_c) + "$" for _c in table.columns])[:-1]}$ 'r'\\''\n' +
        r'\midrule''\n' +
        r'\\''\n'.join([
            "$" + sp.latex(row.name) + "$ & " + " & ".join([
                "$" + sp.latex(sp.nsimplify(row[_c])) + "$"
                for _c in table.columns
            ])
            for i, row in table.iterrows()
        ]) + r'\\' +
        r'\bottomrule''\n'
        r'\end{tabular}''\n'
        r'\end{center}''\n'
        r'\end{table}''\n'
    )


def simplex_shit(table, X, double=False, artificial=False):
    iteration = 0
    while np.any(table.loc['f(x)'][1:] < 0) or double:
        if artificial and table.at['w(x)', 'B'] == 0:
            document.write(
                r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n'
            )
            print_table(table)
            document.write(
                r'Так как $\omega(x)=0$, следовательно, можно исключить $x_8$ из симплекс-таблицы.\\'
                rf'Проверим правильность расчета функции $f(x)={sp.latex(_f)}$.\\'
                r'Так как $x_1,x_2$ являются базисными переменными, необходимо исключить их из значения '
                r'функции, выразив через другие переменные.\\'
                r'Выразим $x_1$:\\'
            )
            x1_row = table.iloc[np.where([table[x1].values != 0, table[x2].values == 0])[0][0]]
            x1_exp = sp.nsimplify(x1_row['B']) - np.sum([
                sp.nsimplify(x1_row[_x]) * _x
                if _x != x1
                else 0
                for _x in X
            ])
            document.write(
                '$$' + sp.latex(sp.Eq(
                    np.sum([sp.nsimplify(x1_row[_x]) * _x for _x in X]),
                    sp.nsimplify(x1_row['B'])
                )) + r'$$\\$$' + sp.latex(sp.Eq(
                    x1,
                    x1_exp
                )) + r'$$\\'
                     r'Выразим $x_2$:'
            )
            x2_row = table.iloc[np.where([table[x2].values != 0, table[x1].values == 0])[0][0]]
            x2_exp = sp.nsimplify(x2_row['B']) - np.sum([
                sp.nsimplify(x2_row[_x]) * _x
                if _x != x2
                else 0
                for _x in X
            ])
            document.write(
                '$$' + sp.latex(sp.Eq(
                    np.sum([sp.nsimplify(x2_row[_x]) * _x for _x in X]),
                    sp.nsimplify(x2_row['B'])
                )) + r'$$\\$$' + sp.latex(sp.Eq(
                    x2,
                    x2_exp
                )) + r'$$\\' +
                r'Выразим $f(x)$:'
                f'$$f(x) = {sp.latex(_f)} = {sp.latex(_f.subs({x2: x2_exp}).subs({x1: x1_exp}))}$$'
            )
            artificial = False
            table = table.drop('w(x)', axis=0) \
                .drop(X[7], axis=1)
            if X[7] in table.index:
                table = table.drop(X[7], axis=0)
            X.pop()

        if double:
            document.write(
                r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n'
            )
            print_table(table)
            document.write(
                r'В симплекс-таблице в столбце базисных переменных есть отрицательные элементы, '
                r'значит используем алгоритм двойственного симплекс-метода.'
            )
            basis = table.iloc[:-1]['B']
            # Находим максимальный по модулю элемент среди отрицательных строк
            leadRow = basis[basis < 0].idxmax()
            leadCol = table.columns[np.argmin([
                abs(table.at[leadRow, 'B'] / table.at[leadRow, _x])
                if table.at[leadRow, _x] < 0
                else np.Infinity
                for _x in X
            ]) + 1]
        else:
            document.write(
                r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n'
            )
            print_table(table)
            document.write(
                r'В симплекс-таблице есть отрицательные коэффициенты строки $f(x)$, '
                r'значит данное базисное решение не оптимально.'
            )
            # Находим ведущий столбец
            leadCol = table.iloc[-1][1:].idxmin()
            # print(table[leadCol])
            # Находим ведущую строку
            leadRow = table.index[np.argmin([
                table.iloc[i]['B'] / row
                if row > 0
                else np.Infinity
                for i, row in enumerate(table[leadCol][:-1])
            ])]
        iteration += 1

        # Находим ведущий элемент
        leadElem = sp.nsimplify(table.at[leadRow, leadCol])
        # Делим элементы главной строки на ведущий элемент
        for col in table.columns:
            table.at[leadRow, col] = sp.Rational(
                sp.nsimplify(table.at[leadRow, col]),
                leadElem
            )
        # Пересчитываем элементы таблицы
        for row in table.index:
            if row == leadRow:
                continue
            aik = sp.nsimplify(table.at[row, leadCol])
            for col in table.columns:
                table.at[row, col] = sp.nsimplify(
                    sp.nsimplify(table.at[row, col]) - sp.nsimplify(table.at[leadRow, col]) * aik
                )
        # Остановка двойного симплекс метода
        if np.all(table['B'] >= 0):
            double = False
        # Обновляем индекс таблицы
        table.index.values[table.index == leadRow] = leadCol
        table = table.reindex()

    document.write(
        r'\subparagraph{Итерация ' + str(iteration) + r'}\mbox{}\\''\n'
    )
    print_table(table)
    document.write(
        r'В симплекс-таблице все коэффициенты строки $f(x)$ неотрицательные, '
        r'значит данное базисное решение оптимально.\\'
        rf'Таким образом, $f^* = f({table.at[x1, "B"]}, {table.at[x2, "B"]}) = {table.at["f(x)", "B"]}$.'
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
    _f = f
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
    y_line = - 5 / 6 * x_line
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
        r'\begin{figure}[!ht]'
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
        r'$$ f^* = f_{max} = '
        f'f({optX[x1]},{optX[x2]}) = {f.subs(optX)} $$'
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
    x_line = -11 * (y_line - 53) / 4 + 80
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
        r'\begin{figure}[!ht]'
        r'\centering'
        r'\includegraphics[height=10cm]{plot_l4_2.png}'
        r'\caption{Новое ограничение}'
        r'\end{figure}'
        r'Уравнение прямой имеет вид: '
        r'$$4 x_1 + 11 x_2 = 783$$'
        r'Ограничение имеет вид: '
        r'$$4 x_1 + 11 x_2 \geq 783$$'
    )
    X.append(sp.symbols('x7'))
    systemX.append(
        sp.Eq(
            -4 * x1 - 11 * x2 + X[6],
            -783
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
    table.iloc[4] = [X[6], -783, -4, -11, 0, 0, 0, 0]
    table = table.append(series, ignore_index=True)
    table[X[6]] = [0, 0, 0, 0, 1, 0]
    table = table.set_index('Базис')
    simplex_shit(pd.DataFrame(table, copy=True, dtype=sp.FractionField), X)

    document.write(
        r'\paragraph{Используем двойственный симплекс-метод для решения задачи}'
        r'\mbox{}\\'
    )
    table = pd.DataFrame(simplexTable, copy=True)
    series = pd.Series(table.iloc[4], copy=True)
    table.iloc[4] = [X[6], -783, -4, -11, 0, 0, 0, 0]
    table = table.append(series, ignore_index=True)
    table[X[6]] = [0, 0, 0, 0, 1, 0]
    table = table.set_index('Базис')
    simplex_shit(pd.DataFrame(table, copy=True, dtype=sp.FractionField), X, double=True)

    document.write(
        r'\paragraph{Используем искусственную переменную для решения задачи}'
        r'\mbox{}\\'
        r'Введем в левую часть ограничения $4 x_1 + 11 x_2 \geq 783$ неотрицательную '
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
    table = pd.DataFrame(simplexTable, copy=True)
    series = pd.Series(table.iloc[4], copy=True)
    table.iloc[4] = [X[7], -783, -4, -11, 0, 0, 0, 0]
    table = table.append(series, ignore_index=True)
    table[X[6]] = [0, 0, 0, 0, 1, 0]
    table = table.set_index('Базис')
    _t = pd.DataFrame(table, copy=True, dtype=sp.FractionField)
    # Меняем строку с x7 на x8
    _t.index.values[_t.index == X[6]] = X[7]
    _t = _t.reindex()
    # Добавляем строку w(x)
    _t.loc['w(x)'] = [783, 4, 11, 0, 0, 0, 0, -1]
    # _t = _t.append(series)
    # Добавляем столбец x8
    _t[X[7]] = [0, 0, 0, 0, 1, 0, 0]
    simplex_shit(_t, X, double=True, artificial=True)
    document.write(r'\end{document}')
    return
