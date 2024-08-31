import numpy as np

ar1 = np.random.randint(0, 10, 20)
print(f'ar1 = {ar1}')
filteredArr = ar1 > 5
print(f'filteredArr = {filteredArr}')
print(f'ar1[filteredArr] = {ar1[filteredArr]}')