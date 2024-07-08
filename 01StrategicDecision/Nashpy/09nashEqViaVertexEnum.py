import numpy as np
import nashpy as nash

A = np.array([[1, 1, -1], [2, -1, 0]])
B = np.array([[.5, -1, -.5], [-1, 3, 2]])
game = nash.Game(A, B)
print(f'Game:\n{game}')

eqs = game.vertex_enumeration()
print(f'List of Equilibrium: {list(eqs)}')
