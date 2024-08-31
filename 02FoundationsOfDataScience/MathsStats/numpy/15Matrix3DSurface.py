import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=plt.figaspect(0.4))
elevation = 30
# azimuth = 0
roll = 0
# # 3D Meshgrid
x_coord = np.arange(0, 10, .1)
y_coord = np.arange(0, 10, .1)
# print(f'x_coord = {x_coord}\ny_coord = {y_coord}')
X_mesh, Y_mesh = np.meshgrid(x_coord, y_coord)
# print(f'X_mesh\n{X_mesh * 0}\nY_mesh\n{Y_mesh}')

Zzero = X_mesh*0
# print(f'Zzero\n{Zzero}')
CONST = 4
Zconst = Zzero + CONST
# print(f'Zconst\n{Zconst}')
# # 2D Matrices
ZslopeX = CONST - X_mesh/CONST
# print(f'ZslopeX\n{ZslopeX}')
ZsinY = np.sin(CONST*Y_mesh)
# print(f'ZsinY\n{ZsinY}')

azimuth = 30
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X_mesh, Y_mesh, Zzero, color='green', alpha=0.2, linewidth=1, label='z=2')
ax.plot_surface(X_mesh, Y_mesh, Zconst, color='pink', alpha=0.2, linewidth=1, label='z=2')
ax.plot_surface(X_mesh, Y_mesh, ZslopeX, color='blue', alpha=0.2, linewidth=1, label='z=2-x/2')
ax.plot_surface(X_mesh, Y_mesh, ZsinY, color='pink', alpha=0.2, linewidth=1, label='z=sin(2y)')
ax.view_init(elevation, azimuth, roll)
plt.title("3D Surfaces")
plt.xlabel("X->")
plt.ylabel("Y->")
# plt.legend(loc='upper right')

azimuth = 60
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X_mesh, Y_mesh, Zzero, color='green', alpha=0.2, linewidth=1, label='z=2')
ax.plot_surface(X_mesh, Y_mesh, Zconst, color='pink', alpha=0.2, linewidth=1, label='z=2')
ax.plot_surface(X_mesh, Y_mesh, ZslopeX, color='blue', alpha=0.2, linewidth=1, label='z=2-x/2')
ax.plot_surface(X_mesh, Y_mesh, ZsinY, color='pink', alpha=0.2, linewidth=1, label='z=sin(2y)')
ax.view_init(elevation, azimuth, roll)
plt.title("3D Surfaces")
plt.xlabel("X->")
plt.ylabel("Y->")
# plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('./plots/15Matrix3DSurface.png')
plt.show()
