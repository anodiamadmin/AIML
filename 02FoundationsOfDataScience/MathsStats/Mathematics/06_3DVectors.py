import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(12)

# 3D Row Vector (3D => [[X, Y, Z]]) represented by Row Array of length = 3
row_vec1 = np.array([[1, 5, 1]])
row_vec2 = np.array([[5, 2, 6]])
# Addition of Row Vectors (2D => [[X, Y]])
sum_vec = row_vec1 + row_vec2

# # Vector Graphics
origin = np.array([[0, 0, 0]])
orig_vec1 = np.concatenate((origin.T, row_vec1.T), axis=1)
orig_vec2 = np.concatenate((origin.T, row_vec2.T), axis=1)
orig_sum_vec = np.concatenate((origin.T, sum_vec.T), axis=1)
print(f'orig_vec1\n{orig_vec1}\nshape = {orig_vec1.shape}')
vec2_vec1 = np.concatenate((row_vec2.T, (row_vec2+row_vec1).T), axis=1)
vec1_vec2 = np.concatenate((row_vec1.T, (row_vec1+row_vec2).T), axis=1)

ax1 = plt.subplot2grid(shape=(1, 2), loc=(0, 0), colspan=1, rowspan=1, projection='3d')
ax1.plot([0, 8], [0, 0], [0, 0], color='red', linewidth=2)
ax1.plot([0, 0], [0, 8], [0, 0], color='red', linewidth=2)
ax1.plot([0, 0], [0, 0], [0, 8], color='red', linewidth=2)
ax1.plot(orig_vec1[0], orig_vec1[1], orig_vec1[2], color='purple', linewidth=1, label='v1=(1,5,1)', marker='o')
ax1.plot(orig_vec2[0], orig_vec2[1], orig_vec2[2], color='green', linewidth=1, label='v2=(5,2,6)', marker='o')
ax1.view_init(5, 40, 0)
ax1.set_title('3D Vectors (1D Array of length=3)')
ax1.set_xlabel("<-- X")
ax1.set_ylabel("Y -->")
ax1.set_zlabel("< --Z")
ax1.legend(loc='upper right')
ax1.grid(True)

ax2 = plt.subplot2grid(shape=(1, 2), loc=(0, 1), colspan=1, rowspan=1, projection='3d')
ax2.plot([0, 8], [0, 0], [0, 0], color='red', linewidth=2)
ax2.plot([0, 0], [0, 8], [0, 0], color='red', linewidth=2)
ax2.plot([0, 0], [0, 0], [0, 8], color='red', linewidth=2)
ax2.plot(orig_vec1[0], orig_vec1[1], orig_vec1[2], color='purple', linewidth=1, marker='o')
ax2.plot(orig_vec2[0], orig_vec2[1], orig_vec2[2], color='green', linewidth=1, marker='o')
ax2.plot(orig_sum_vec[0], orig_sum_vec[1], orig_sum_vec[2], color='blue', linewidth=2, label='Resultant', marker='o')
ax2.plot(vec2_vec1[0], vec2_vec1[1], vec2_vec1[2], color='purple', alpha=0.3, linewidth=1, label='v1', marker='o')
ax2.plot(vec1_vec2[0], vec1_vec2[1], vec1_vec2[2], color='green', alpha=0.3, linewidth=1, label='v2', marker='o')
ax2.view_init(5, 40, 0)
ax2.set_title('Resultant of 3D Vector in space')
ax2.set_xlabel("<-- X")
ax2.set_ylabel("Y -->")
ax2.set_zlabel("< --Z")
ax2.legend(loc='upper right')
ax2.grid(True)

plt.show()
plt.tight_layout()
plt.savefig(f'./plots/06_3DVectors.png')
