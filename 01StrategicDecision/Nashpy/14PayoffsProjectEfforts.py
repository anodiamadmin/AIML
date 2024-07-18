import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash

ax = plt.axes(projection='3d')

y_pts = np.linspace(0, 10, 101)
p_pts = np.linspace(0, 10, 101)
y_mesh, p_mesh = np.meshgrid(y_pts, p_pts)
payoff_y = 60 * y_mesh * p_mesh - y_mesh**3
payoff_p = 60 * y_mesh * p_mesh - 8 * p_mesh**3

ax.plot(y_mesh, p_mesh, payoff_y, color='green', alpha=0.2, linewidth=.5, label=f'Payoff of Y')
ax.plot(y_mesh, p_mesh, payoff_p, color='red', alpha=0.2, linewidth=.5, label=f'Payoff of P')

y_max_p_pts = np.sqrt(20 * p_pts)
y_max_vals = 60 * y_pts * y_max_p_pts - y_pts**3

p_max_p_pts = p_pts**2 * 2 / 5
p_max_vals = 60 * p_max_p_pts * y_pts - 8 * p_max_p_pts**3

ax.plot(y_pts, y_max_p_pts, y_max_vals, color='green', linewidth=1, label='y_max_wrt_p')
ax.plot(y_pts, p_max_p_pts, p_max_vals, color='red', linewidth=1, label='p_max_wrt_y')

p_same_payoff_y = y_pts / 2
same_payoff = 60 * y_pts * p_same_payoff_y - y_pts ** 3
ax.plot(y_pts, p_same_payoff_y, same_payoff, color='blue', linewidth=1, label='y_p_same_payoff')

ax.plot(1, p_pts, 60 * 1 * p_pts - 1**3, color='green', linewidth=.4)
ax.plot(2, p_pts, 60 * 2 * p_pts - 2**3, color='green', linewidth=.4)
ax.plot(3, p_pts, 60 * 3 * p_pts - 3**3, color='green', linewidth=.4)
ax.plot(4, p_pts, 60 * 4 * p_pts - 4**3, color='green', linewidth=.4)
ax.plot(5, p_pts, 60 * 5 * p_pts - 5**3, color='green', linewidth=.4)
ax.plot(6, p_pts, 60 * 6 * p_pts - 6**3, color='green', linewidth=.4)
ax.plot(7, p_pts, 60 * 7 * p_pts - 7**3, color='green', linewidth=.4)
ax.plot(8, p_pts, 60 * 8 * p_pts - 8**3, color='green', linewidth=.4)
ax.plot(9, p_pts, 60 * 9 * p_pts - 9**3, color='green', linewidth=.4)
ax.plot(10, p_pts, 60 * 10 * p_pts - 10**3, color='green', linewidth=.4)

ax.plot(y_pts, 1, 60 * 1 * y_pts - 8 * 1**3, color='red', linewidth=.4)
ax.plot(y_pts, 2, 60 * 2 * y_pts - 8 * 2**3, color='red', linewidth=.4)
ax.plot(y_pts, 3, 60 * 3 * y_pts - 8 * 3**3, color='red', linewidth=.4)
ax.plot(y_pts, 4, 60 * 4 * y_pts - 8 * 4**3, color='red', linewidth=.4)
ax.plot(y_pts, 5, 60 * 5 * y_pts - 8 * 5**3, color='red', linewidth=.4)
ax.plot(y_pts, 6, 60 * 6 * y_pts - 8 * 6**3, color='red', linewidth=.4)
ax.plot(y_pts, 7, 60 * 7 * y_pts - 8 * 7**3, color='red', linewidth=.4)
ax.plot(y_pts, 8, 60 * 8 * y_pts - 8 * 8**3, color='red', linewidth=.4)
ax.plot(y_pts, 9, 60 * 9 * y_pts - 8 * 9**3, color='red', linewidth=.4)
ax.plot(y_pts, 10, 60 * 10 * y_pts - 8 * 10**3, color='red', linewidth=.4)

game = nash.Game(payoff_y, payoff_p)
for nash_equilibrium in next(game.support_enumeration()):
    if nash_equilibrium is None:
        break
    else:
        print(nash_equilibrium)
        print(len([nash_equilibrium][0]))
# print(f'nash_equilibrium({type(nash_equilibrium)})\n{nash_equilibrium}')
# next_nash = next(nash_equilibrium)
# print(f'next_nash({type(next_nash)})\n{next_nash}')

ax.legend(loc='upper right')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(-6000, 4000)
plt.title("3D Surface Plots")
plt.xlabel("y -->")
plt.ylabel("p -->")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.savefig('./image/14PayoffsProjectEfforts.png')
plt.show()
