import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x_val = np.linspace(-2, 2, 101)
# print(f'x = {x_val}')

# Create the plots
y_val0 = x_val**2
# print(f'y = {y_val1}')
plt.plot(x_val, y_val0, label="y=x^2", c='grey', alpha=0.5, linewidth=1)

y_val1 = x_val**3
# print(f'y = {y_val1}')
plt.plot(x_val, y_val1, label="y=x^3", c='purple', alpha=0.5, linewidth=1)

y_val2 = 3*x_val**2 + 0.5
plt.plot(x_val, y_val2, label="y=3(x^2)+.5", c='green', alpha=0.5, linewidth=1, linestyle='-')

y_val3 = 2 * np.sqrt(x_val) + 1
plt.plot(x_val, y_val3, label="2sqrt(x)+1", c='#ff9944', alpha=0.5, linewidth=1.5, linestyle='--')

y_val4 = -(x_val**2) + 2
plt.plot(x_val, y_val4, label="-(x^2)+2", c='black', alpha=0.5, linewidth=1, linestyle=':')

y_val3 = -2.5 * np.sqrt(x_val-.75) - 3
plt.plot(x_val, y_val3, label="-2.5*sqrt(x-.75)+3", c='blue', alpha=0.5, linewidth=1, linestyle='--')

plt.title("Straight Lines")
plt.xlabel("X -->")
plt.ylabel("Y -->")
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(-5, 5)
plt.axhline(0,color='red', linewidth=1.5) # x = 0
plt.axvline(0,color='red', linewidth=1.5) # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/16ParabolaEquation.png')
plt.show()