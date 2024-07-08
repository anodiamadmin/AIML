import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# define derivative numerically
def dx(x, t, A):
    f = np.dot(A, x)
    phi = np.dot(x, f)
    return x * (f - phi)


t = np.linspace(0, 20, 101)
# t = np.linspace(0, 10, 101)
# A = np.array([[4, 3, 2], [2, 1, 5], [6, 0, 3]])
A = np.array([[0, 3, 2], [2, 1, 5], [6, 0, 3]])
xs = odeint(func=dx, y0=[1/3, 1/3, 1/3], t=t, args=(A,))

plt.plot(xs)
plt.show()
