import numpy as np
'''
+,-,*,/,%,**,1/
'''
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
CONST = 10
arr3 = arr1 + CONST
print(f'arr3 = arr1 + {CONST} = {arr3}')
arr4 = arr1 * arr2
print(f'arr4 = arr1 * arr2 = {arr4}')
'''
np.add(), np.subtract(), np.multiply(), np.divide(), np.mod(), np.power(), np.reciprocal()
'''
arr5 = np.array([1.3, 2.2, 1.3, 0.4])
arr6 = np.array([2.5, 1.6, 3.1, 0.8])
CONST = 10
arr7 = np.subtract(arr5, CONST)
print(f'arr7 = np.subtract(arr5, CONST) = {arr7}')
arr8 = np.power(arr5, arr6)
print(f'arr8 = np.power(arr5, arr6) = {arr8}')
arr9 = np.reciprocal(arr5)
print(f'arr9 = np.reciprocal(arr5) = {arr9}')

# 2D Arrays
arr10 = np.array([[13, 12, 10], [14, 9, 16]])
arr11 = np.array([[3, 2, 4], [6, 5, 3]])
arr12 = np.mod(arr10, arr11)
print(f'arr12 = np.mod(arr10, arr11) =\n{arr12}')
arr13 = np.array([[13.0, 12.2, 10.1], [14, 9, 16]])
arr14 = np.array([3, 2, 4])
arr15 = arr13 % arr14
print(f'arr15 = arr10 % arr14 =\n{arr15}')
