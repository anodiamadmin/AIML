import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(6, 6))

x_val = np.linspace(-2, 2, 11)
print(f'X Array = {x_val}')
# y_val = np.linspace(-.5, 3.5, 11)

# Create the plots
y_val1 = .5*x_val + 2
print(f'Y Array = {y_val1}')
plt.plot(x_val, y_val1, label="y=5x+2", c='purple', marker='o', alpha=0.5, linewidth=1)
plt.fill_between(x_val, 3.5, y_val1, color='purple', alpha=0.2, label='y>=5x+2')

y_val2 = x_val
plt.plot(x_val, y_val2, label="y=x", c='green', alpha=0.5, linewidth=1, linestyle=':')
plt.fill_between(x_val, y_val2, -0.5, label="y<=x", color='green', alpha=0.1)


plt.title("Straight Line In-equations")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.ylim(-0.5, 3.5)
plt.xlim(-2, 2)
plt.axhline(0, color='red', linewidth=1.5)   # x axis
plt.axvline(0, color='red', linewidth=1.5)   # y axis
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/01StLineInEquation.png')
plt.show()