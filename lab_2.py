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
        ','.join([sp.latex(x) for x in X[0:2]]) +
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
        '$x^*=(%d, %d), f(x^*)=%d$' % (optX[x1], optX[x2],
                                       f.subs({x1: optX[x1], x2: optX[x2]}))
    )
    document.write(
        '\n\nТеорема о дополнительной нежесткости:\n\n'
    )
    theorem = []
    for i, s in enumerate(systemY):
        theorem.append(sp.Equality(
            sp.Mul(optX[X[i]], (s.lhs - s.rhs), evaluate=False), 0))
    for i, s in enumerate(systemX):
        left = Y[i] * sp.Add(s.lhs.subs({p: optX[p]
                                         for p in X[0:2]}), -s.rhs, evaluate=False)
        theorem.append(sp.Equality(left, 0, evaluate=False))

    theoremY = theorem[2:]
    _y = []
    __y = []
    for i, s in enumerate(theoremY):
        if s.doit() == True:
            _y.append(Y[i])
        else:
            __y.append(Y[i])

    document.write(
        r'$\begin{cases}' +
        r',\\'.join([sp.latex(s) for s in theorem]) +
        r'.\end{cases}  \Rightarrow' +
        r'\begin{cases}' +
        r',\\'.join([sp.latex(_) + r' ^ * \neq 0 ' for _ in _y]) + r',\\' +
        r',\\'.join([sp.latex(_) + '^* = 0' for _ in __y]) +
        r'.\end{cases}$'
    )

    document.write(
        '\n\nНайдем значения $' +
        '$ и $'.join([sp.latex(e)+'^*' for e in _y]) + '$:\n\n'
    )

    resultY = {p: 0 for p in __y}
    solveY = [sp.Equality(
        s.lhs.subs(resultY),
        s.rhs
    ) for s in systemY]
    resultY.update(dict(zip([e for e in _y], *sp.linsolve(solveY, _y))))
    document.write(
        r'$$\begin{cases}' +
        r',\\'.join([sp.latex(sp.Equality(s.lhs, s.rhs)) for s in systemY]) +
        r'.\end{cases}' +
        r'\Rightarrow \begin{cases}' +
        r',\\'.join([sp.latex(s) for s in solveY]) +
        r'.\end{cases}' +
        r'\Rightarrow \begin{cases}' +
        r',\\'.join([sp.latex(sp.Equality(p, resultY[p])) for p in _y]) +
        r'.\end{cases} \Rightarrow ' +
        r'y^*(' +
        r','.join([sp.latex(resultY[sp.symbols('y%d' % (i + 1))]) for i in range(len(resultY))]) +
        r')\\$$' + '\n\n'
    )
    document.write(
        r'$$\varphi(y^*)=' +
        sp.latex(phi) + '=' +
        sp.latex(phi.subs({p: sp.symbols(str(resultY[p])) for p in resultY})) + '=' +
        sp.latex(phi.subs(resultY)) +
        r'\\$$'
    )
    if phi.subs(resultY) == f.subs(optX):
        document.write(
            '\n\nЗначения совпали:' +
            r'$f(x^*)=\varphi(y^*)={0}$'.format(phi.subs(resultY))
        )
    else:
        document.write('<h3 style="color: crimson">Соси член, кусок долбаеба.\
        Какую-то хуйню сделал... Ищи ошибку!</h3>')
