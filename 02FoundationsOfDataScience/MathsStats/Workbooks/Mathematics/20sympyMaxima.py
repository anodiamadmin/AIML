import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x = np.linspace(-5, 5, 101)
y = -x**2 + 2*x + 2
dx = .1
dy_dx = np.gradient(y)
maxima = np.argwhere(np.absolute(dy_dx) < dx**3)
# print(f'maxima = {maxima}')
# print(f'maxima_x = {x[maxima][0]} :: maxima_y = {y[maxima][0]} maxima_dy_dx = {dy_dx[maxima]}')
# print(f'np.array([x[maxima-3], x[maxima+3]]) = {x[maxima-3][0], x[maxima+3][0]}')
# Create the plots
plt.plot(x, y, label="y=-x^2+2x+2", c='purple', alpha=0.5, linewidth=1)
plt.plot(x, dy_dx, label="y=dy_dx", c='green', alpha=0.5, linewidth=1)
plt.plot((x[maxima][0]-3*dx, x[maxima][0]+3*dx), [y[maxima][0], y[maxima][0]],
         label="y=maxima", c='blue', alpha=.8, linewidth=1)
plt.title("Polynomial equations and Standalone Derivatives")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.axhline(0, color='red', linewidth=1.5)  # x = 0
plt.axvline(0, color='red', linewidth=1.5)  # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/20sympyMaxima.png')
plt.show()
