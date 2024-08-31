import numpy as np

# i-integer, u-unsigned integer, f-float, c-complex, m-timedelta, M-datetime
# arr8 = np.array([7, 8, 9, 10], dtype="f")
# bool_, int_, intp, intc, int8, int16, int32, int64, uint8, uint16, uint32, uint64
# float_, float16, float32, float64, complex_, complex64, complex128
arr8 = np.array([7, 8, 9, 10, 0], dtype=np.float32)
print(f'arr8.dtype = {arr8.dtype}')

arr9 = np.int8(arr8)
print(f'arr9.dtype = {arr9.dtype}')

arr10 = np.array([7.8, 9, 10, 0])
print(f'arr10.dtype = {arr10.dtype}')

arr11 = np.int8(arr10)
print(f'arr11.dtype = {arr11.dtype} :: arr11 = {arr11}')

arr12 = arr11.astype(bool)
print(f'arr12.dtype = {arr12.dtype} :: arr12 = {arr12}')
