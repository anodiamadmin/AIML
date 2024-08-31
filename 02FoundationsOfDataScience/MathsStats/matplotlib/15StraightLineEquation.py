import numpy as np
from matplotlib import pyplot as plt

# plt.style.use('fivethirtyeight')
plt.figure(figsize=(6, 6))

x_val = np.linspace(-2, 2, 11)
print(f'x_val = {x_val}')

# Create the plots
y_val1 = .5*x_val + 2
print(f'y_val = {y_val1}')
plt.plot(x_val, y_val1, label="y=.5x+2", c='purple', marker='o', alpha=0.5, linewidth=1)

# y_val2 = x_val
# plt.plot(x_val, y_val2, label="y=x", c='green', alpha=0.5, linewidth=1, linestyle=':')
#
# y_val3 = np.ones(11) * 0.75
# # print(f'ones = {np.ones(11)}')
# # print(f'y_val3 = {y_val3}')
# plt.plot(x_val, y_val3, label="y=0.75", c='brown', alpha=0.5, linewidth=1, linestyle='-')
#
# y_val4 = -0.5*x_val + 1
# plt.plot(x_val, y_val4, label="y=0.5x - 1", c='blue', alpha=0.5, linewidth=1, linestyle='--')
#
# plt.axvline(1.25, label="x=1.25", c='black', alpha=0.5, linewidth=1, linestyle='--')

plt.title("Straight Lines")
plt.xlabel("X -->")
plt.ylabel("Y -->")
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(-0.5, 3.5)
plt.axhline(0,color='red', linewidth=1.5) # x = 0
plt.axvline(0,color='red', linewidth=1.5) # y = 0
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/15SStraightLineEquation.png')
plt.show()