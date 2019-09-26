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
    r'\usepackage{graphicx}''\n'
    r'\usepackage{fontspec}''\n'
    r'\setmainfont{Monofur Nerd Font}''\n'

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
            ' & '.join([sp.latex(column) for column in row.A1])
            if isinstance(row, np.matrix)
            else str(row)
            for row in matrix
        ]) +
        r'\end{matrix}''\n'
        r'\right)'
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
    def iterate(Xb, Cb, B, not_basis_index):
        print('B = ', B)
        print('Cb = ', Cb)
        invB = np.linalg.inv(B)
        Y = Cb * invB
        print('Y = ', Y)
        NB = np.matrix([
            [
                s.lhs.coeff(X[i])
                for i in not_basis_index
            ]
            for s in systemX
        ], dtype='float')
        c = [f.coeff(X[i]) for i in not_basis_index]
        print('NB = ', NB)
        new_basis = Y * NB - c
        document.write(
            r'\paragraph{Итерация 1}\mbox{}\\''\n'
            r'\subparagraph{Шаг 1} - Вычисление $z_j - с_j$ для небазисных векторов '
            '$P_1$  и $P_2$.''\n'
            r'$$Y=C_B \cdot B^{-1}=' +
            print_matrix(Cb) + ' \cdot ' +
            print_matrix(invB) + ' = ' +
            print_matrix(Y) +
            r',$$\\''\n'
            r'$$(z_1 - c_1, z_2 - c_2)=Y \cdot (P_1,P_2) - (c_1,c_2)=' +
            print_matrix(Y) + r' \cdot ''\n' +
            print_matrix(NB) + ' - ''\n' +
            print_matrix(c) + ' = ' +
            print_matrix(new_basis) + '$$''\n'
        )
    Xb = X[2:]
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
