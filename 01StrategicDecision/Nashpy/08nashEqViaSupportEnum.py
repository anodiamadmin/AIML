import numpy as np
import nashpy as nash

A = np.array([[1, 1, -1], [2, -1, 0]])
B = np.array([[.5, -1, -.5], [-1, 3, 2]])
game = nash.Game(A, B)
print(f'Game:\n{game}')

eqs = game.support_enumeration()
print(f'Iterating the type {type(eqs)}')
for eq in eqs:
    print(f'eq: {eq}')
print(f'Iterating can only be done once over type {type(eqs)}')
for eq in eqs:
    print(f'\neq: {eq}')

eqs = game.support_enumeration()
print(f'List of Equilibrium: {list(eqs)}')
