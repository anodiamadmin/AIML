import numpy as np


def myfunc(a, b):
    if a > b:
        return a - b
    else:
        return a + b


vfunc = np.vectorize(myfunc)
vecArr = vfunc([1, 2, 3, 4, 3, 1, 12, 45], 2)
print(f'Vectorized Function returned array = {vecArr}')
