import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig.set_figheight(5)
fig.set_figwidth(9)

# # 2D Matrix with linearly increasing values
diagonal_array = np.diag(range(5))
print(f'diagonal_array\n{diagonal_array}')
ax[0][0].matshow(diagonal_array)
# ax[0][0] = sns.heatmap(diagonal_array, annot=True, fmt="d", cbar=False)
ax[0][0].set_title("2D Diagonal Array")
ax[0][0].set_xlabel("Columns")
ax[0][0].set_ylabel("Rows")
ax[0][0].set_xlim(0, 5)
ax[0][0].set_ylim(5, 0)
ax[0][0].set_xticks(ticks=range(0, 6, 1))
ax[0][0].set_xticks(ticks=range(0, 6, 1))
#
# zeros_array = np.zeros((4, 3),  dtype=int)
# print(f'zeros_array\n{zeros_array}')
# ax[0][1].matshow(zeros_array)
# ax[0][1].set_title("2D Zero Array")
#
# ones_array = np.ones((4, 3),  dtype=int)
# print(f'ones_array\n{ones_array}')
# ax[0][2].matshow(ones_array)
# ax[0][2].set_title("2D Ones Array")
#
# eye_array = np.eye(5, 3)
# print(f'eye_array\n{eye_array}')
# ax[1][0].matshow(eye_array)
# ax[1][0].set_title("2D Eye Array")
#
# full_array = np.full((5, 5), 10)
# print(f'full_array\n{full_array}')
# ax[1][1].matshow(full_array)
# ax[1][1].set_title("2D Full Array")
#
# rand_array = np.random.randint(0,  10, (5, 4))
# print(f'rand_array\n{rand_array}')
# ax[1][2].matshow(rand_array)
# ax[1][2].set_title("2D Random Array")

plt.grid(True)
plt.tight_layout()
plt.show()