import numpy as np

arr1 = np.random.randint(0, 10, 12)
print(f'Origianl Array = {arr1}')

arrAcs = np.sort(arr1)
print(f'Sorted Array = {arrAcs}')

arrDsc = np.sort(arr1)[::-1]
print(f'Reverse Sorted Array = {arrDsc}')

print(f'Only copy is sorted. Original Array remains unchanged: {arr1}')

arr3D = arr1.reshape(2, 2, 3)
print(f'3D Origianl Array =\n{arr3D}')

arrAcs3D = np.sort(arr3D)
print(f'Sorted 3D Array =\n{arrAcs3D}')

arrDsc3D = np.sort(arr3D)[::-1]
print(f'Reverse Sorted Array =\n{arrDsc3D}')

arrStr = np.array(["Jade", "Ruth", "Alina", "Soffia", "Amaya", "Diya", "Anoushka", "Tom"])
print(f'Origianl String Array = {arrStr}')

arrStrDcs = np.sort(arrStr)[::-1]
print(f'Descending String Array = {arrStrDcs}')

arrBol = np.array([False, True, True, False, False, False, True, True, False, False, True])
print(f'Origianl Boolean Array = {arrBol}')

arrBolAcs = np.sort(arrBol)
print(f'Ascending Boolean Array = {arrBolAcs}')
