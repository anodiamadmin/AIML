import numpy as np

# https://numpy.org/doc/stable/user/basics.broadcasting.html

# CONST broadcast over (n,) array
CONST = 3
ary1 = np.array([3, 2, 3, 1, 2])
print(f'Const {CONST} broadcast over {np.shape(ary1)} array {ary1}')
print(f'{CONST} * {ary1} = {CONST * ary1}')

# CONST broadcast over (n,m) matrix
ary2 = np.array([[3, 2], [3, 1], [2, 1]])
print(f'Const {CONST} broadcast over {np.shape(ary2)} array')
print(f'[[3,2],[3,1],[2,1]] ^ 3 =\n{ary2 ** CONST}')

# CONST broadcast over (n, m, p) array
ary3 = np.array([[[3, 2], [3, 1]], [[2, 1], [1, 2]]])
print(f'Const {CONST} broadcast over {np.shape(ary3)} array')
print(f'3/[[[3,2],[3,1]],[[2,1],[1,2]]] =\n{CONST/ary3}')

# (n,) 1D array broadcast over (n, m, p) array
ary1D = np.array([4, 6])
print(f'(n,) array {ary1D} broadcast over {np.shape(ary3)} array')
print(f'{ary1D}*[[[3,2],[3,1]],[[2,1],[1,2]]] =\n{ary1D*ary3}')

# (n,1) Row Vector broadcast over (n, m, p) array
rowV = np.array([[5, 4]])
print(f'(n,) array {rowV} broadcast over {np.shape(ary3)} array')
print(f'{rowV}+[[[3,2],[3,1]],[[2,1],[1,2]]] =\n{rowV+ary3}')

# (1, n) Column Vector broadcast over (n, m, p) array
colV = np.array([[3], [2]])
print(f'(n,) array {colV} broadcast over {np.shape(ary3)} array')
print(f'[[[3,2],[3,1]],[[2,1],[1,2]]] - [[3][2]] =\n{ary3 - colV}')
