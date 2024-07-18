import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash

ax = plt.axes(projection='3d')
ax.view_init(19, 16, 0)

y_pts = np.linspace(0, 15, 151)
p_pts = np.linspace(0, 15, 151)
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
ax.plot(y_pts, p_same_payoff_y, same_payoff, color='purple', linewidth=1, label='y_p_same_payoff')

for pt in range(0, 16):
    print(f'pt = {pt}')
    ax.plot(pt, p_pts, 60 * pt * p_pts - pt**3, color='green', linewidth=.4)
    ax.plot(y_pts, pt, 60 * y_pts * pt - 8*pt**3, color='red', linewidth=.4)

ax.plot([0, 0], [0, 0], [-7000, 7000], color='orange', linewidth=2, label='Nash Eq: 1')
ax.plot([5, 5], [10, 10], [-7000, 7000], color='blue', linewidth=2, label='Nash Eq: 2')
ax.scatter(0, 0, 0 ,color='orange', marker='*', s=100, edgecolors='k')
ax.scatter(5, 10, (60*5*10-5**3), color='green', marker='*', s=100, edgecolors='k')
ax.scatter(5, 10, (60*5*10-8*10**3), color='red', marker='*', s=100, edgecolors='k')

# game = nash.Game(payoff_y, payoff_p)
# for nash_equilibrium in next(game.support_enumeration()):
#     if nash_equilibrium is None:
#         break
#     else:
        # print(nash_equilibrium)
        # print(len([nash_equilibrium][0]))
# print(f'nash_equilibrium({type(nash_equilibrium)})\n{nash_equilibrium}')
# next_nash = next(nash_equilibrium)
# print(f'next_nash({type(next_nash)})\n{next_nash}')

ax.legend(loc='upper right')
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_zlim(-7000, 7000)
plt.title("3D Surface Plots")
plt.xlabel("y -->")
plt.ylabel("p -->")
plt.legend(loc='upper left', fontsize=6)
plt.grid(True)
plt.tight_layout()

plt.savefig('./image/14PayoffsProjectEfforts.png')
plt.show()
