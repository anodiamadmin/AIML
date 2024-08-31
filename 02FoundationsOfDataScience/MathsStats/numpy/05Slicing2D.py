import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
fig.set_figheight(5)
fig.set_figwidth(10)

# # 2D Matrix with linearly increasing values
array2d = np.arange(400).reshape(20, 20)
for row in range(len(array2d)):
    for col in range(len(array2d[0])):
        if array2d[row][col] % 7 == 0:
            array2d[row][col] = 0
        elif array2d[row][col] % 7 == 4:
            array2d[row][col] = 400
clipped2D = array2d[4:16, 3:17]
shortend2D = array2d[0:20:2, 0:20:2]

ax[0].matshow(array2d)
ax[0].set_title("2D Array")
ax[0].set_xlabel("Columns")
ax[0].set_ylabel("Rows")
ax[0].set_xlim(-0.5, 19.5)
ax[0].set_ylim(19.5, -0.5)
ax[0].set_xticks(ticks=range(0, 20, 5))
ax[0].set_xticks(ticks=range(0, 20, 5))

ax[1].matshow(clipped2D)
ax[1].set_title("Clipped from all sides")

ax[2].matshow(shortend2D)
ax[2].set_title("Shortened Array")

plt.grid(True)
plt.tight_layout()
plt.show()