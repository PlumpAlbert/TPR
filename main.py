#!/usr/bin/python3

import os
from functools import reduce
from math import gcd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

import lab_1
import lab_2

os.system('reset')

x1, x2 = sp.symbols("x1 x2")
f: sp.Add = 5 * x1 + 6 * x2
# Исходные точки
points = [
    {x1: 0, x2: 80},
    {x1: 90, x2: 80},
    {x1: 80, x2: 60},
    {x1: 55, x2: 20},
    {x1: 25, x2: 0}
]

# Лабораторная работа № 1

systemX, X, optX = lab_1.main(x1, x2, points, f)

# Конец ЛР 1

# Лабораторная работа № 2

lab_2.main(f, systemX, X, optX)

# Конец ЛР 2


print('Finish!')
