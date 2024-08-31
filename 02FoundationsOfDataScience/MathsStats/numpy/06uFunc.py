import numpy as np

# np.min(), np.max(), np.sum(),  np.prod(),  np.mean(), np.std(), np.var(), np.median(), np.argmin()
# np.argmax(), np.round(), np.floor(), np.ceil(), np.histogram(), np.histogram2d()
# np.cumsum(),  np.cumprod(), np.diff(), np.gradient(), np.angle()
# np.sqrt(), np.cbrt(), np.square(), np.reciprocal(), np.divide()
# np.sin(), np.cos(), np.tan(), np.arcsin(), np.arccos(), np.arctan(), np.sinh(), np.cosh(), np.tanh()
# np.log(), np.log10(), np.log2(), np.exp(), np.expm1(), np.power(), np.mod()

arr1 = np.array([[5, 2, -4, 1, -3], [-2, -4, 7, 6, 1]])
print(f'np.min(arr1) = {np.min(arr1)}')
print(f'np.mean(arr1) = {np.mean(arr1)}')
print(f'np.var(arr1) = {np.var(arr1)}')
print(f'np.std(arr1) = {np.std(arr1)}')
print(f'np.prod(arr1, axis=0) = {np.prod(arr1, axis=0)}')
print(f'np.cumsum(arr1, axis=1) = {np.cumsum(arr1, axis=1)}')
print(f'np.argmin(arr1, axis=1) = {np.argmin(arr1, axis=1)}')
print(f'np.gradient(arr1, axis=1) = {np.gradient(arr1, axis=1)}')
print(f'np.sigmoid(arr1) = {1/(1+np.exp(-arr1))}')
