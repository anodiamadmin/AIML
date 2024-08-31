import numpy as np
from matplotlib import pyplot as plt
import math

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(11, 6))
x_val = np.linspace(0, 1000, 500001)[1:]
y_val1 = np.log(x_val)
y_val2 = np.log10(x_val)
y_val3 = np.log2(x_val)

ax[0][0].set_title('Logarithm')
ax[0][0].plot(x_val, y_val1, label="y=ln(x)", c='green', alpha=0.5, linewidth=1, linestyle='-')
ax[0][0].plot(x_val, y_val2, label="y=log10(x)", c='red', alpha=0.5, linewidth=1, linestyle='-')
ax[0][0].plot(x_val, y_val3, label="y=l0g2(x)", c='blue', alpha=0.5, linewidth=1, linestyle='-')
ax[0][0].set_xlabel("x ->")
ax[0][0].set_ylabel("y ->")
ax[0][0].legend(loc="lower right")

ax[1][0].plot(x_val, y_val1, label="y=ln(x)", c='green', alpha=0.5, linewidth=1, linestyle='-')
ax[1][0].plot(x_val, y_val2, label="y=log10(x)", c='red', alpha=0.5, linewidth=1, linestyle='-')
ax[1][0].plot(x_val, y_val3, label="y=l0g2(x)", c='blue', alpha=0.5, linewidth=1, linestyle='-')
ax[1][0].legend(loc="lower right")
ax[1][0].set_xlabel("10^x ->")
ax[1][0].set_ylabel("y ->")
ax[1][0].set_xscale('log')

x_val1 = np.linspace(-5, 5, 101)
y_val4 = np.exp(x_val1)
y_val5 = np.power(2, x_val1)
y_val6 = 50 / (1 + np.exp(-x_val1))

ax[0][1].set_title('Exponential')
ax[0][1].plot(x_val1, y_val4, label="y=e^x", c='green', alpha=0.5, linewidth=1, linestyle='-')
ax[0][1].plot(x_val1, y_val5, label="y=2^x", c='red', alpha=0.5, linewidth=1, linestyle='-')
ax[0][1].plot(x_val1, y_val6, label="y=50/1+e^-x", c='blue', alpha=0.5, linewidth=1, linestyle='-')
ax[0][1].set_xlabel("x ->")
ax[0][1].set_ylabel("y ->")
ax[0][1].legend(loc="upper left")

ax[1][1].plot(x_val1, y_val4, label="y=e^x", c='green', alpha=0.5, linewidth=1, linestyle='-')
ax[1][1].plot(x_val1, y_val5, label="y=2^x", c='red', alpha=0.5, linewidth=1, linestyle='-')
ax[1][1].plot(x_val1, y_val6, label="y=50/1+e^-x", c='blue', alpha=0.5, linewidth=1, linestyle='-')
ax[1][1].legend(loc="lower right")
ax[1][1].set_xlabel("x ->")
ax[1][1].set_ylabel("10^y ->")
ax[1][1].set_yscale('log')

plt.tight_layout()
plt.savefig('./plots/04Log_Exponential.png')
plt.show()
