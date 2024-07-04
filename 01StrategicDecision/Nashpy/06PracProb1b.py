import numpy as np
import nashpy as nash

y_effort = np.array([0, 1, 2])
p_effort = np.array([0, 1, 2])


def calc_return(y, p):
    return 2*y + 2*p + 0.5*y*p - y**2, 2*y + 2*p + 0.5*y*p - p**2


y_effort_mesh, p_effort_mesh = np.meshgrid(y_effort, p_effort)
y_return, p_return = calc_return(y_effort_mesh, p_effort_mesh)
print(f'Y return\n{y_return}\nP return\n{p_return}')

game = nash.Game(y_return, p_return)
nash_equilibrium = game.support_enumeration()
print(f'nash_equilibrium({type(nash_equilibrium)})\n{nash_equilibrium}')
next_nash = next(nash_equilibrium)
print(f'next_nash({type(next_nash)})\n{next_nash}')
