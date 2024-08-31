import numpy as np

arr1 = np.array([1, 2, 3, 4, 5, 6])
print(arr1)
# Insert 50 at position 2
arr2 = np.insert(arr1, 2, 50)
print(arr2)
# Insert 40 at position (2, 4)
arr3 = np.insert(arr1, (2, 4), 40)
print(arr3)
# Float cannot be inserted to int array, so it is floored to int, Trying to insert String gives error, Bool can be entered though
arr4 = np.insert(arr1, (3, 3), 92.879)
print(arr4)
# Append
arr5 = np.append(arr1, [7, 8, 9])
print(arr5)
# 2D Arrays
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
# Insert 50 at position 2
arr2d1 = np.insert(arr2d, 2, 50, axis=1)
print(arr2d1)
arr2d2 = np.insert(arr2d, (0, 2), [65, 7, 35], axis=0)
print(arr2d2)
# Delete
del2D = np.delete(arr2d2, 1, axis=1)
print(del2D)