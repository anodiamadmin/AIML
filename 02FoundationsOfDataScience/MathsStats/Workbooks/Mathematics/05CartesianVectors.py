import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(12)

# 2 D Row Vector (2D => [[X, Y]]) represented by Row Array of length = 2
ary = np.array([-3, 5])
print(f'1D Array = {ary} :: shape = {ary.shape}')
row_vector1 = ary.reshape(1, len(ary))
print(f'Row Vector 1 = {row_vector1} :: shape = {row_vector1.shape}')

# Vector Graphics
origin = np.array([[0, 0]])
viz_row_vector1 = np.concatenate((origin, row_vector1), axis=0).T
# print(f'viz_row_vector1\n{viz_row_vector1}')
ax1 = plt.subplot2grid(shape=(1, 3), loc=(0, 0), colspan=1, rowspan=1)
ax1.plot(viz_row_vector1[0], viz_row_vector1[1], label="(-3,5)", c='purple', alpha=0.5, linewidth=1, marker='o')
ax1.set_title('Row Vector')
ax1.set_xlabel("X -->")
ax1.set_ylabel("Y -->")
ax1.set_ylim(-2, 9)
ax1.set_xlim(-4, 5)
ax1.axhline(0, color='red', linewidth=1.5)  # x Axis
ax1.axvline(0, color='red', linewidth=1.5)  # y Axis
ax1.legend(loc='lower right')
ax1.grid(True)

# 2 D Column Vector (2D => [[X], [Y]]) represented by Column Array
column_vector = np.array([[4], [2]])
print(f'Column Vector\n{column_vector}\nv1.shape = {column_vector.shape}')

# Vector Graphics
viz_col_vector = np.concatenate((origin.T, column_vector), axis=1)
# print(f'viz_col_vector\n{viz_col_vector}')
ax2 = plt.subplot2grid(shape=(1, 3), loc=(0, 1), colspan=1, rowspan=1)
ax2.plot(viz_col_vector[0], viz_col_vector[1], label="(4,2)", c='purple', alpha=0.5, linewidth=1, marker='o')
ax2.set_title('Column can also be Vector')
ax2.set_xlabel("X -->")
ax2.set_ylabel("Y -->")
ax2.set_ylim(-2, 9)
ax2.set_xlim(-4, 5)
ax2.axhline(0, color='red', linewidth=1.5)  # x Axis
ax2.axvline(0, color='red', linewidth=1.5)  # y Axis
ax2.legend(loc='lower right')
ax2.grid(True)

row_vector2 = column_vector.T
print(f'Row Vector 2 = {row_vector2} :: shape = {row_vector2.shape}')

row_vector_sum = row_vector1 + row_vector2
print(f'Row Vector Sum = {row_vector_sum} :: shape = {row_vector_sum.shape}')

# Addition of Row Vectors (2D => [[X, Y]])
row_vec1 = np.array([[-3, 5]])
row_vec2 = np.array([[4, 2]])
sum_vec = row_vec1 + row_vec2
# Graphics
viz_row_vec1 = np.concatenate((origin, row_vec1), axis=0).T
viz_row_vec2 = np.concatenate((origin, row_vec2), axis=0).T
reflect_v2 = np.concatenate((row_vec1, sum_vec), axis=0).T
viz_sum_vec = np.concatenate((origin, sum_vec), axis=0).T
# print(f'sum_vec\n{sum_vec}')
ax3 = plt.subplot2grid(shape=(1, 3), loc=(0, 2), colspan=1, rowspan=1)
ax3.plot(viz_row_vec1[0], viz_row_vec1[1], label="(-3,5)", c='purple', alpha=0.5, linewidth=1, marker='o')
ax3.plot(viz_row_vec2[0], viz_row_vec2[1], label="(4,2)", c='purple', alpha=0.5, linewidth=1, marker='o')
ax3.plot(reflect_v2[0], reflect_v2[1], label="(4,2)", c='purple', alpha=0.5, linewidth=1, marker='o', linestyle=':')
ax3.plot(viz_sum_vec[0], viz_sum_vec[1], label="sum", c='blue', alpha=0.5, linewidth=2, marker='o',)
ax3.set_title('Addition of Vectors')
ax3.set_xlabel("X=x1+x2 ->")
ax3.set_ylabel("Y=y1+y2 ->")
ax3.set_ylim(-2, 9)
ax3.set_xlim(-4, 5)
ax3.axhline(0, color='red', linewidth=1.5)  # x Axis
ax3.axvline(0, color='red', linewidth=1.5)  # y Axis
ax3.legend(loc='upper right')
ax3.grid(True)

plt.show()
plt.tight_layout()
plt.savefig(f'./plots/05CartesianVectors.png')
