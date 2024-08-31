import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

x0 = np.random.normal(10, 5, 40)
y0 = 2.5*np.random.normal(10, 5, 40)
plt.scatter(x0, y0, color="#fd7c59", label="Low/ Random")

x1 = np.random.normal(10, 5, 20)
y1 = x1 + np.random.normal(0, 1, 20)
plt.scatter(x1, y1, color="#9ca9cf", label="High1")

x2 = np.random.normal(10, 5, 20)
y2 = 2*x2 - 3 + np.random.normal(1, 1, 20)
plt.scatter(x2, y2, color="#9cf3a9", label="High2")

x3 = np.random.normal(10, 5, 20)
y3 = x3 + 2*x3 - 4 + np.random.normal(0, 5, 20)
plt.scatter(x3, y3, color="#592359", label="Medium")

plt.title("Scatter Plot for Correlation")
plt.xlabel("X-values")
plt.ylabel("Y-values")

plt.xlim(0, 20)
plt.ylim(0, 40)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/13Correlation.png')
plt.show()