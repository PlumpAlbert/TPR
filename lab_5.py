#!/usr/bin/python3

import sympy as sp
import numpy as np
import pandas as pd

document = open('lab_5.tex', 'w')
document.write(
    r'\documentclass[14pt,a4paper]{report}''\n'
    r'\renewcommand{\figurename}{Рис.}''\n'
    r'\usepackage{amsmath}''\n'
    r'\usepackage{booktabs}''\n'
    r'\usepackage[english,russian]{babel}''\n'
    r'\usepackage{fontspec}''\n'
    r'\usepackage{mathptmx}''\n'
    r'\defaultfontfeatures{Ligatures={TeX},Renderer=Basic}''\n'
    r'\setmainfont[Ligatures={TeX,Historic}]{Times New Roman}''\n'
    r'\usepackage{graphicx}''\n'

    r'\title{Лабораторная работа №5}''\n'
    r'\author{Plump Albert}''\n'

    r'\begin{document}''\n'
    r'\begin{titlepage}''\n'
    r'\maketitle''\n'
    r'\newpage''\n'
    r'\end{titlepage}''\n'
)


def print_matrix(matrix):
    return (
        r'\left(''\n'
        r'\begin{matrix}''\n' +
        r' \\'.join([
            ' & '.join([sp.latex(sp.nsimplify(column)) for column in row.A1])
            if isinstance(row, np.matrix)
            else sp.latex(sp.nsimplify(row))
            for row in matrix
        ]) +
        r'\end{matrix}''\n'
        r'\right)'
    )


def print_table(X, Xb, fx, alpha_inv, xb, lead_column_index):
    return (
        r'\centering''\n'
        '\n'r'\begin{tabular}[c!]{|c|'+ f'{"".join(["|c" for _x in X])}' + '|c|}\n'
        r'\toprule''\n'
        'Базис & ' +
        f'{" & ".join(["$"+sp.latex(_x)+"$" for _x in X])}' +
        r' & Решение \\''\n'
        r'\midrule''\n'
        '$f(x)$ & '+f'{" & ".join([f"${sp.nsimplify(c)}$" for c in fx])}'+r' & \\' +
        r"\\".join([
            "$" + sp.latex(_x) + "$ & " + " & ".join([
                "$" + sp.latex(sp.nsimplify(alpha_inv.item((i, 0)))) + "$"
                if j == lead_column_index
                else " "
                for j, __x in enumerate(X)
            ]) + f" & ${sp.nsimplify(xb.item((i, 0)))}$"
            for i, _x in enumerate(Xb)
        ]) + r'\\''\n'
        r'\bottomrule''\n'
        r'\end{tabular}'
    )



def main(
        X,
        systemX,
        f,
        optX
):
    systemX = systemX[:-1]
    X = X[:-1]
    document.write(
        r'\section{Задание}''\n'
        r'\begin{enumerate}''\n'
        r'\item Решить задачу линейного программирования, используя итерации '
        'модифицированного симплекс-метода.''\n'
        r'\item Вычислить коэффициенты z-строки и определить включаемую в базис'
        'переменную $x_j$.''\n'
        r'\item Определить исключаемую переменную''\n'
        r'\item Определить новый базис и перейти к шагу 2''\n'
        r'\end{enumerate}''\n'
        r'\section{Решение}''\n'
        r'\begin{figure}[h!]''\n'
        r'\centering''\n'
        r'\includegraphics[height=10cm]{plot_l4.png}''\n'
        r'\caption{Область ограничений}''\n'
        r'\end{figure}''\n'

        rf'Целевая функция имеет вид: $f(x)={sp.latex(f)} \rightarrow max$.\\''\n'
        'А область ограничений задачи в стандартной форме имеет вид:''\n'
        r'\begin{equation*}''\n'
        r'X = \begin{cases}''\n' + 
        r',\\''\n'.join([sp.latex(s).replace('=', '&=') for s in systemX]) + r'\\' +
        ','.join([sp.latex(x) for x in X]) +
        r'&\geq 0.''\n'
        r'\end{cases}''\n'
        r'\end{equation*}''\n'
        'Найденное оптимальное решение в предыдущих практических работах:''\n'
        '$$f^*=f_{max}=f(100,80)=1220$$''\n'
        'Начальное решение:''\n'
        '$$X_B=(x_3,x_4,x_5,x_6)^T,$$\\''\n'
        '$$C_B=(0,0,0,0),$$''\n'
        '$$B=(P_3,P_4,P_5,P_6)=I,$$''\n'
        '$$B^{-1}=I.$$'
    )
    def iterate(Xb, Cb, B, not_basis_index, iteration):
        B_inv = np.linalg.inv(B)
        Y = Cb * B_inv
        P = np.matrix([
            [
                s.lhs.coeff(X[i])
                for i in not_basis_index
            ]
            for s in systemX
        ], dtype='float')
        c = [f.coeff(X[i]) for i in not_basis_index]
        new_basis = Y * P - c
        fx = [
            new_basis.item(not_basis_index.index(i))
            if i in not_basis_index
            else 0
            for i, _x in enumerate(X)
        ]
        lead_column_index = np.argmin(new_basis)
        document.write(
            r'\subparagraph{Шаг 1} - Вычисление $z_j - c_j$ для небазисных '
            'векторов $P_1$  и $P_2$.''\n'
            r'$$Y=C_B \cdot B^{-1}=' +
            print_matrix(Cb) + ' \cdot ' +
            print_matrix(B_inv) + ' = ' +
            print_matrix(Y) +
            r',$$\\''\n'
            rf'$$({",".join([f"z_{i+1} - c_{i+1}" for i in not_basis_index])})'
            f'=Y \cdot ({",".join([f"P_{i+1}" for i in not_basis_index])})'
            f'- ({",".join([f"c_{i+1}" for i in not_basis_index])})=' +
            print_matrix(Y) + r' \cdot ''\n' +
            print_matrix(P) + ' - ''\n' +
            print_matrix(c) + ' = ' +
            print_matrix(new_basis) + '$$''\n'
            'Следовательно, включению в базис подлежит вектор '
            rf'$P_{lead_column_index}$.\\'
        )

        b = np.matrix([[s.rhs] for s in systemX])
        xb = B_inv * b
        alpha_inv = B_inv * P[:,lead_column_index]
        lol = xb / np.where(alpha_inv > 0, alpha_inv, 0)
        lead_row_index = np.argmin(lol)
        document.write(
            r'\subparagraph{Шаг 2} - Определение исключаемого вектора при '
            f'введении в базис вектора $P_{lead_column_index}$.''\n'
            r'$$X_B=B^{-1} \cdot b=' +
            print_matrix(B_inv) + r' \cdot ' +
            print_matrix(b) + ' = ' +
            print_matrix(xb) + r',$$\\'

            rf'$$\alpha^{iteration}'r'=B^{-1} \cdot 'f'P_{lead_column_index}=' +
            print_matrix(B_inv) + r' \cdot ' +
            print_matrix(P[lead_column_index]) + ' = ' +
            print_matrix(alpha_inv) + r'.$$\\' +
            print_table(X, Xb, fx, alpha_inv, xb, lead_column_index) +
            'Отсюда следует, что $\theta = min(' +
            ",".join([
                sp.nsimplify(r)
                if r != np.Inf
                else continue
                for r in np.nditer(lol)
            ]) + f')={np.min(lol)}.'r'$\\''\n'
            'Значит, исключению из базиса подлежит вектор $P_{lead_row_index}$.'
        )
        
       zeta = alpha_inv / -alpha_inv.item(lead_row_index)
       zeta.item(lead_row_index) = 1.0 / alpha_inv.item(lead_row_index)
        document.write(
            r'\subparagraph{Шаг № 3} - определение обратной матрицы, '
            r'соответствующей новому базису.\\'
            f'Так как вместо вектора $P_{lead_row_index}$ в базис вводится '
            f'вектор $P_{lead_column_index}$ '
            rf'и $\alpha^{iteration}=' + 
            print_matrix(alpha_inv.transpose()) + '^T$, то:'
        )
    Xb = X[2:]
    document.write('\paragraph{Итерация № 1}\mbox{}\\''\n')
    iterate(
        Xb,
        np.matrix([f.coeff(_x) for _x in Xb], dtype='float'),
        np.matrix([
            [s.lhs.coeff(_x) for s in systemX]
            for _x in Xb
        ], dtype='float'),
        [0,1]
    )
    document.write(r'\end{document}')
