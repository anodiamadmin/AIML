import numpy as np

arr1 = np.random.randint(1, 10, (2, 3, 4))
print(f'3D ARRAY = \n{arr1}\n')

print(f'ITERATING -->')
for i in range(arr1.shape[0]):
    for j in range(arr1.shape[1]):
        for k in range(arr1.shape[2]):
            print(f'Page={i}, Row={j}, Col={k} :: VALUE = {arr1[i, j, k]}')

print(f'\nnp.nditer() -->')
for x in np.nditer(arr1):
    print(f'VALUE = {x}')