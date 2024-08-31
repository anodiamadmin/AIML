import numpy as np

mat1 = np.array([[1, 2, 3], [4, 5, 6]])
mat2 = np.array([[2, 3, 1], [4, 5, 0], [1, 6, 5]])

# ***** Transpose
print(f'Transpose of Matrix 1:\n{mat1.T}')
print(f'Transpose of Transpose:\n{mat1.T.transpose()}')

# ***** Swapaxes
arr4D = np.arange(24).reshape(2, 3, 4)
print(f'4D-Array:\n{arr4D}')
print(f'Swapaxes:\n{arr4D.swapaxes(0, 2)}')

# Inverse Matrix
print(f'Inverse of Matrix 2:\n{np.linalg.inv(mat2)}')

# Power of Matrix
# Determinate of Matrix
