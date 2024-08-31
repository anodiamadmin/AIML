import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=plt.figaspect(0.3))
elevation = 30
azimuth = 30
roll = 0
# # 3D Meshgrid
x_coord = np.arange(0, 10, .1)
y_coord = np.arange(0, 5, .1)
X, Y = np.meshgrid(x_coord, y_coord)

# 2D Matrix Z1
Z1 = 20-(Y+X)
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_wireframe(X, Y, Z1, color='blue', alpha=0.2, linewidth=1, label='z1=20-x-y')
ax.view_init(elevation, azimuth, roll)
plt.title("Matrix")
plt.xlabel("<-- X")
plt.ylabel("Y -->")
plt.legend(loc='upper right')

# Matrix Arith Constant/ Array
CONST = 2
arr = np.sin(x_coord)
# print(arr)
Z2 = Z1 + arr + CONST
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_wireframe(X, Y, Z1, label='z=z1', color='green', alpha=0.1)
surf = ax.plot_surface(X, Y, Z2, label='z=z1+sin(x)+2', cmap='plasma')
ax.view_init(elevation, azimuth, roll)
plt.title("Matrix - Constant/ Array")
plt.xlabel("<-- X")
plt.ylabel("Y -->")
plt.legend(loc='upper left')
fig.colorbar(surf, shrink=0.5, aspect=10)

# Matrix Arith Matrix
Z3 = Z2/Z1
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(X, Y, Z3, label='(1+sin(x)+z2)/z1', cmap='plasma')
ax.view_init(elevation, azimuth, roll)
plt.title("Matrix - Matrix")
plt.xlabel("<-- X")
plt.ylabel("Y -->")
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./plots/16MatrixArith.png')
plt.show()