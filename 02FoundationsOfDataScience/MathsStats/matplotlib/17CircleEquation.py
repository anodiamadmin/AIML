import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x_val = np.linspace(-2, 2, 100)

# Create the plots
rad = 1
y_val1 = np.sqrt(rad**2 -(x_val**2))
plt.plot(x_val, y_val1, label="r=1, (0,0)", c='purple', alpha=0.5, linewidth=1)

rad = 2
y_val2 = np.sqrt(rad**2 -(x_val**2))
plt.plot(x_val, y_val2, label="r=2, (0,0)", c='#ff9944', alpha=0.5, linewidth=1)

rad = .5
y_val2 = -np.sqrt(rad**2 -(x_val**2)) - .5
plt.plot(x_val, y_val2, label="r=.5, (0,-.5)", c='green', alpha=0.5, linewidth=1)

plt.title("Straight Lines")
plt.xlabel("X -->")
plt.ylabel("Y -->")
# plt.xscale('log')
# plt.yscale('log')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.axhline(0,color='red', linewidth=1.5) # x = 0
plt.axvline(0,color='red', linewidth=1.5) # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/17CircleEquation.png')
plt.show()