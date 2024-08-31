import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

azimuthL = 10
azimuthR = 12

def plot_3d_cardboard_box(eye, azimuth):
    # fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection='3d')

    # Single Point
    ax.scatter3D(30, 50, 250, color='red', marker='o')

    # Random Points
    # x_rand = np.random.randint(0, 100, size=10)
    # y_rand = np.random.randint(0, 100, size=10)
    # z_rand = np.random.randint(0, 400, size=10)
    # ax.scatter3D(x_rand, y_rand, z_rand, color='blue', marker='v', alpha=0.5)

    # Parabola (3D)
    x_coord = np.arange(0, 100, 0.1)
    y_coord = np.arange(0, 100, 0.1)
    z_prbla = x_coord * y_coord / 25
    ax.plot(x_coord, y_coord, z_prbla, color='green', alpha=0.5, linewidth=1, label='z=xy/25')

    # Wave (3D)
    z_wave = np.sin(x_coord) * np.cos(y_coord) * 200 + 200
    ax.plot(x_coord, y_coord, z_wave, color='pink', linewidth=1, label='z=200+200sin(x)cos(y)')

    # # ax.set_title("3D Cardboard Box")
    # ax.set_xlabel("X -->")
    # ax.set_ylabel("Y -->")
    # # ax.legend(loc='upper left')

    elevation = 30
    roll = 0
    ax.view_init(elevation, azimuth, roll)

    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./plots/32CardboardBox3D{eye}.png')
    plt.show()


plot_3d_cardboard_box('Left', azimuthL)
plot_3d_cardboard_box('Right', azimuthR)
print(f"COMPLETE - SUCCESS")
