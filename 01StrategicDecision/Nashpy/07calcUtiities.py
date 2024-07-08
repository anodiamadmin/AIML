import numpy as np
import nashpy as nash

print('\n----- Matching Pennies Game -----')
A = [[1, -1], [-1, 1]]
B = [[-1, 1], [1, -1]]
matching_pennies_game = nash.Game(A, B)
print(f'Matching Pennies Game:\n{matching_pennies_game}')
print('\n----- Utilities Score .8/.2 vs .6/.4-----')
sigma_r = np.array([.8, .2])  # row player's strategy .8 of the times 1st row, .2 of the times second row
sigma_c = np.array([.6, .4])  # col player's strategy .6 of the times 1st col, .4 of the times second col
print(f'Matching Pennies (.8/.2 vs .6/.4) Utilities Score:\n{matching_pennies_game[sigma_r, sigma_c]}')
print('\n----- Utilities Score 1./.0 vs .6/.4-----')
sigma_r = np.array([1., .0])  # row player's strategy 1. of the times 1st row, .0 of the times second row
print(f'Matching Pennies (1./.0 vs .6/.4) Utilities Score:\n{matching_pennies_game[sigma_r, sigma_c]}')
