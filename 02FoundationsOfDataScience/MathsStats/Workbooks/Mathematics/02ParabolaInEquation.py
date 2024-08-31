import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x_val = np.linspace(-2, 2, 51)

# Create the plots
y_val1 = x_val**2
print(f'x = {x_val}')
print(f'y = {y_val1}')
plt.plot(x_val, y_val1, label="y=x^2", c='purple', alpha=0.5, linewidth=1)
plt.fill_between(x_val, -0.5, y_val1, color='purple', alpha=0.2, label='y<=x^2')

y_val2 = 3*x_val**2 + 0.5
plt.plot(x_val, y_val2, label="y=3(x^2)+.5", c='green', alpha=0.5, linewidth=1, linestyle='-')
plt.fill_between(x_val, y_val2, 4.5, color='green', alpha=0.2, label='y>=3(x^2)+.5')

plt.title("Parabola In-equations")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.ylim(-0.5, 4.5)
plt.axhline(0,color='red', linewidth=1.5) # x = 0
plt.axvline(0,color='red', linewidth=1.5) # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/02ParabolaInEquation.png')
plt.show()
