# $>> pip install sympy
# https://colab.research.google.com/drive/1l6FrbsJi5yWrDQ05ViI2QhXiSjcp5vt7
import sympy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

from math import cos

# INTRODUCTION
print(f'Sympy Version = {sp.__version__}')
rt2 = sp.sqrt(2)                 # SYMBOLIC:: graphic O/P in google colab\n
rt2_math = math.sqrt(2)          # NUMERIC::
print(f'rt2 = {rt2} ')
print(f'rt2^2 in Sympy = {rt2 ** 2}\nrt2^2 in Math = {rt2_math ** 2}\n')

# SYMBOLS
x = sp.Symbol('x')
print(f'x = {x}')               # Graphic O/P in google colab
exp = 2 * x + 5
print(f'exp = {exp}')           # Graphic O/P in google colab
print(f'{sp.sin(x)**2 + sp.cos(x)**2}')
print(f'{2*x/6}')
print(f'{x/x}')
expr = x * (x + 1)
print(f'expr = {expr} :: expr.expand = {expr.expand()}')         # Graphic O/P in google colab
my_var = sp.symbols('price')
print(f'my_var = {my_var}')     # Graphic O/P in google colab
double_price = my_var * 2
print(f'double_price = {double_price}')  # Graphic O/P in google colab

my_symbols = sp.symbols('x y z t')
print(f'my_symbols = {my_symbols}, type(my_symbols) = {type(my_symbols)}')
a, b, c = sp.symbols('a b c', real=True)
print(f'a = {a}, type(a) = {type(a)}, b = {b}, type(b) = {type(b)},  c = {c}, type(c) = {type(c)}')
poly = a*(b + c)**2
print(f'sp.expand(poly) = {sp.expand(poly)}\n')          # Graphic O/P in google colab

# FACTORIZE
expr = x**2 + 2*x - 15
print(f'sp.factor(expr) = {sp.factor(expr)}')           # Graphic O/P in google colab
print(f'expr.factor() = {expr.factor()}')           # Graphic O/P in google colab

# VECTOR SYMBOLS
x_vector = sp.symbols('x0:10')
print(f'x_vector = {x_vector}')                      # Graphic O/P in google colab
print(f'x_vector[3]**2 + sp.ln(x_vector[7]) = '
      f'{x_vector[3]**2 + sp.ln(x_vector[7])}')  # Graphic O/P in google colab

# f1x = 4*ln(x) - x       # Q 1
# f1x = 20*x - 2*x**2     # Q 2
# f1x = -5*x              # Q 3
f1x = 10*x - x**2 - 4   # Q 4



# DERIVATIVES
df1x_dx = sp.diff(f1x, x)
print(f'df1x_dx = {df1x_dx}')
