import numpy as np

arr1 = np.array([[2, 3, 2, 4, 5], [3, 2, 1, 4, 2]])

# Unique
print(f'np.unique = {np.unique(arr1)}')

# Search
print(f'np.where(arr1>2) = {np.where((arr1[1]) > 2)} :: {np.where((arr1[1]) > 2)[0]}')

print(arr1)
# Sort
print(f'np.sort(arr1) =\n{np.sort(arr1, axis=1)}')
# Descending Sort (Assignment to correct this)
print(f'descending np.sort(arr1) =\n{np.sort(arr1, axis=1)[::-1]}')

# Filter
filter = (arr1[0] > 2)
print(f'filter = {filter}')
print(f'filtered arr1[1] = {arr1[1][filter]}')

# Shuffle array
arr = np.array([1, 2, 3, 4, 5])
print("Original array: ", arr)
np.random.shuffle(arr)
print(f"Shuffled array: {arr}")

# Flatten Array Order = C (row major, Default), F (column major)
arr2 = np.array([1, 2, 3, 4, 5, 6, 3, 4, 1, 2, 5, 6])
arr3d = np.resize(arr2, (4, 3))
print(f'arr3d =\n{arr3d}')
# flat = arr3d.flatten() # order='C'
flat = arr3d.flatten(order='F')
# flat = np.ravel(arr3d, order='C') #order='F'
print(f'flat/raveled = {flat}')