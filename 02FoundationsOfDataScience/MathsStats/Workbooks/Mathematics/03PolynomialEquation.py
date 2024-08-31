import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x_val = np.linspace(-3, 4, 51)

# Create the plots
y_val1 = 2 * x_val**2 - 6 * x_val + 3
# print(f'x = {x_val}')
# print(f'y = {y_val1}')
plt.plot(x_val, y_val1, label="y=2x^2-6x+3", c='purple', alpha=0.5, linewidth=1)

y_val2 = x_val**3 - 4*x_val**2 + 2*x_val - 4
plt.plot(x_val, y_val2, label="y=x^3-4x^2+2x-4", c='green', alpha=0.5, linewidth=1, linestyle='-')

y_val3 = x_val**3
plt.plot(x_val, y_val3, label="y=x^3", c='blue', alpha=0.5, linewidth=1, linestyle='-')

plt.title("Polynomial Equations")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.ylim(-40, 40)
plt.axhline(0, color='red', linewidth=1.5)  # x Axis
plt.axvline(0, color='red', linewidth=1.5)  # y Axis
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/03PolynomialEquation.png')
plt.show()