import numpy as np

mat1 = np.array([[1, 2, 3], [4, 5, 6]])
mat2 = np.array([[2, 3], [4, 5], [1, 6]])
matmul = np.matmul(mat1, mat2)
dotProd = np.dot(mat1, mat2)
print(f'Matrix 1 of shape {mat1.shape}\n{mat1}\nMatrix 2 of shape {mat2.shape}\n{mat2}')
# Matmul # best suited for PRECISION and complex calculations
print(f'Mat Mul of shape:{matmul.shape}\n{matmul}')
#  Dot # efficient for vector operations and well-suited for scenarios where SPEED is essential
print(f'Dot Product of shape:{dotProd.shape}\n{dotProd}')