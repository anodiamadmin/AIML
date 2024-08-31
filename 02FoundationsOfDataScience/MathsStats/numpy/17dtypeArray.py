import numpy as np

# bool_, int_, intp, intc, int8, int16, int32, int64, uint8, uint16, uint32, uint64
# float_, float16, float32, float64, complex_, complex64, complex128

arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f'arr1.dtype = {arr1.dtype}')

arr2 = np.array(["Hello", "World", "from", "Anodiam"])
print(f'arr2.dtype = {arr2.dtype}')

arr3 = np.array([True, True, False, False, True, False])
print(f'arr3.dtype = {arr3.dtype}')

arr4 = np.array([1.319, 2.304, -6.053, -3.221, 0.4625, 2.216, -1.127, 0.028, -2.921, 1.042])
print(f'arr4.dtype = {arr4.dtype}')

arr5 = np.array([1, 2, 3, 4.0, False])
print(f'arr5.dtype = {arr5.dtype}')

arr6 = np.array([1, 2, 3, 4.0, False, 'A'])
print(f'arr6.dtype = {arr6.dtype}')

arr7 = np.array([7, 8, 9, 10], dtype=np.int8)
print(f'arr7.dtype = {arr7.dtype}')

# i-integer, u-unsigned integer, f-float, c-complex, m-timedelta, M-datetime
# arr8 = np.array([7, 8, 9, 10], dtype="f")

# bool_, int_, intp, intc, int8, int16, int32, int64, uint8, uint16, uint32, uint64
# float_, float16, float32, float64, complex_, complex64, complex128
arr8 = np.array([7, 8, 9, 10], dtype=np.float32)
print(f'arr8.dtype = {arr8.dtype}')