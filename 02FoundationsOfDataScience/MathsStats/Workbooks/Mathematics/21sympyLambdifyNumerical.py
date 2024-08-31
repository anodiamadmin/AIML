import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Slow: sp.subs() & sp.evalf() [same as sp.N()] :: FAST: lambdify(points)
x = sp.symbols('x')
points = np.linspace(-5, 5, 101)

y = sp.pi * x**2 + 5 * x + sp.E
lambdify_y = sp.lambdify(x, y, 'numpy')(points)

# print(f'y(1) = {y.subs(x, 1)}')
# print(f'y(1.5) = {y.subs(x, 1.5)}')
# print(f'y(1.5) = {y.subs(x, 1.5).evalf()}')
# print(f'y(1.5) = {y.subs(x, 1.5).evalf(8)}')
# print(f'y(1.5) = {sp.N(y.subs(x, 1.5), 10)}')

# lambdify(points)
dy_dx = y.diff(x)
# print(f'dy_dx = {dy_dx}')
lambdify_dy_dx = sp.lambdify(x, dy_dx, 'numpy')(points)
# print(f'lambdify_dy_dx = {lambdify_dy_dx}')

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

plt.plot(points, lambdify_y, label="y=pi*x^2+5x+e", c='purple', alpha=0.5, linewidth=1)
plt.plot(points, lambdify_dy_dx, label="y=dy_dx", c='green', alpha=0.5, linewidth=1)
plt.title("Lambdify Numerical")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.axhline(0, color='red', linewidth=1.5)  # x = 0
plt.axvline(0, color='red', linewidth=1.5)  # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/21sympyLambdifyNumerical.png')
plt.show()
