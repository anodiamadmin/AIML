import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('fivethirtyeight')
ax = plt.axes(projection='3d')

# # 3D Meshgrid
# x_coord = np.arange(0, 4, 1)
# y_coord = np.arange(0, 4, 1)
# X, Y = np.meshgrid(x_coord, y_coord)
# print(f"X=\n{X}\nShape(X)={X.shape}")
# print(f"Y=\n{Y}\nShape(Y)={Y.shape}")
#
# # 3D Meshgrid Parabola
# Z = X * Y
# print(f"Z=\n{Z}\nShape(Z)={Z.shape}")
# ax.plot_surface(X, Y, Z, color='green', alpha=0.5, linewidth=1, label='z=xy')

# 3D Meshgrid Wave
x_coord = np.arange(-5, 5, .1)
y_coord = np.arange(-5, 5, .1)
X, Y = np.meshgrid(x_coord, y_coord)
# Z = np.ones((10, 10)) * 3
# print(f'Z = {Z}')
# ax.plot_surface(X, Y, Z)
# ax.plot_wireframe(X, Y, Z)
Z = np.sin(X) * np.cos(Y)
ax.plot_surface(X, Y, Z, label='z=sin(X).cos(Y)', cmap='plasma')
elevation = 30
azimuth = 30
roll = 0
ax.view_init(elevation, azimuth, roll)

plt.title("3D Surface Plots")
plt.xlabel("X -->")
plt.ylabel("Y -->")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/19_3DSurfacePlot.png')
plt.show()