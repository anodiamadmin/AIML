import numpy as np

ar1 = np.random.randint(0, 10, 20)
print(f'ar1 = {ar1}')
oneIndex = np.where(ar1 == 1)
print(f'One Index = {oneIndex}, Position of Ones = {oneIndex[0]}')
print(f'Values at {oneIndex[0]} are {ar1[oneIndex[0]]}')

print(f'\nEVEN Items')
evenIndex = np.where(ar1 % 2 == 0)
print(f'Even Index = {evenIndex}, Position of Even = {evenIndex[0]}')
print(f'Values at {evenIndex[0]} are {ar1[evenIndex[0]]}')

print(f'\nODD Items')
oddIndex = np.where(ar1 % 2 == 1)
print(f'Odd Index = {oddIndex}, Position of Odd = {oddIndex[0]}')
print(f'Values at {oddIndex[0]} are {ar1[oddIndex[0]]}')