import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash


def calc_compensation(q1, q2):
    if q1 > q2:
        return 0
    else:
        return q2 + 5


fig = plt.figure(figsize=plt.figaspect(0.8))
fig.suptitle(f'Continuous Strategy Compensation Game', fontsize=12)
ax = fig.add_subplot(1, 1, 1, projection='3d')

q1 = np.linspace(5, 20, 151)
q2 = np.linspace(5, 20, 151)
q1_mesh, q2_mesh = np.meshgrid(q1, q2)
v_calc_compensation = np.vectorize(calc_compensation)
compensation_1 = v_calc_compensation(q1_mesh, q2_mesh)
compensation_2 = v_calc_compensation(q2_mesh, q1_mesh)

ax.plot_surface(q1_mesh, q2_mesh, compensation_1, color='red', alpha=0.2, linewidth=.5, label=f'Player - 1')
ax.plot_surface(q1_mesh, q2_mesh, compensation_2, color='green', alpha=0.2, linewidth=.5, label=f'Player - 2')

# ax.plot3D(q1, [5] * len(q1), [10] * len(q1), color='red', linewidth=1, label='Player-1 Safe Claim')
# ax.plot3D([5] * len(q1), q2, [10] * len(q1), color='green', linewidth=1, label='Player-2 Safe Claim')
# game = nash.Game(q1_mesh, q2_mesh)
# nash_equilibrium = game.support_enumeration()
# next_nash = next(nash_equilibrium)
# print(f'nash_equilibrium({type(nash_equilibrium)})\n{nash_equilibrium}')
# print(f'next_nash({type(next_nash)})\n{next_nash}')

ax.legend(loc='upper right')
ax.set_xlim(-0, 25)
ax.set_ylim(-0, 25)
ax.set_zlim(-5, 30)
ax.set_xlabel('Claim 1 -->')
ax.set_ylabel('Claim 2 -->')
ax.set_zlabel('Compensation -->')

fig.savefig(f'./plots/05ContinuousStrategyGame.png')
plt.show()
