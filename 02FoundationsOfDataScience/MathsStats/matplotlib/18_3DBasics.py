import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
plt.style.use('fivethirtyeight')
ax = plt.axes(projection='3d')

# Single Point
ax.scatter3D(30, 50, 250, color='red', marker='o')
# ax.scatter3D(35, 60, 200, color='blue', marker='d')

# # Random Points
x_rand = np.random.randint(0, 100, size=30)
y_rand = np.random.randint(0, 100, size=30)
z_rand = np.random.randint(0, 400, size=30)
# print(f'x_rand = {x_rand}\ny_rand = {y_rand}\nz_rand = {z_rand}')
ax.scatter3D(x_rand, y_rand, z_rand, color='blue', marker='d', alpha=0.5)

# # Parabola (3D)
x_coord = np.arange(0, 100, 0.1)
# print(f'x = {x_coord}')
y_coord = np.arange(0, 100, 0.1)
# print(f'Y = {y_coord}')
z_prbla = x_coord * y_coord / 25
# print(f'z = {z_prbla}')

ax.plot(x_coord, y_coord, z_prbla, color='green', alpha=0.5, linewidth=1, label='z=xy/25')

# # Wave (3D)
z_wave = np.sin(x_coord*3) * np.cos(y_coord/3) * 200 + 200
ax.plot(x_coord, y_coord, z_wave, color='pink', linewidth=1, label='z=200+200sin(x)cos(y)')

# ax.view_init(45, 60, 90)
plt.title("3D Matplotlib Presentation")
plt.xlabel("X -->")
plt.ylabel("Y -->")
ax.set_zlabel("Apples -->")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/18_3DBasics.png')
plt.show()