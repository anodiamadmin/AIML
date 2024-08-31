import numpy as np

orgl_arr = np.array([5, 4, 3, 2, 1, 0])
print(f'Original array = {orgl_arr}')
# COPY creates a SEPARATE COPY
copy_arr = orgl_arr.copy()
print(f'Copy creates a SEPARATE COPY = {copy_arr}')
copy_arr[3] = 10
print(f'Original array unchanged if Copy is changed = {orgl_arr}')

# VIEW creates a REFERENCE, not a separate copy
view_arr = orgl_arr.view()
print(f'View refers to the same array = {view_arr}')
view_arr[3] = 10
print(f'Original array gets changed if view is changed = {orgl_arr}')

