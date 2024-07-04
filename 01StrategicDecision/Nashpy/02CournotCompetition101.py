import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash
import os
import glob

files = glob.glob('./plots/02Cournot_*.png')
for f in files:
    os.remove(f)

a_crn_const = int(input('Enter Cournot\'s Constant \'a\' ::\n'
                        '\tmoderately large +ve Int (e.g. 100 to 10000 ) : '))
print(f'We will first draw Cournot\'s Competition equation p(q) = {a_crn_const} - q '
      f'where p(q) = Unit price of q identical quantities produced by 2 firms, F1 &\n\tF2 '
      f'and q = q1 + q2 respectively. We will show 3 consideration where p(q) and q are '
      f'equal to ({a_crn_const*.25}, {a_crn_const*.75}), ({a_crn_const*.5}, {a_crn_const*.5})'
      f' and ({a_crn_const*.75}, {a_crn_const*.25}).\n\tFor each of the above, we will consider'
      f' the below 3 values of unit cost of production c1 by firm F1.'
      f' Which will be 25%, 50% and 75% of the unit price p(q).')
print(f'Next, we will draw profits of the 2 firms against their production quantities.')
print(f'Finally, we will draw the NASH EQUILIBRIUM between the 2 firms against their production'
      f' quantities, as per below parameters.')
q_nash = int(input(f'Enter Total Quantity for drawing Nash Equilibrium ::\n'
                   f'\t+ve Int less than {a_crn_const} (q<=a) : '))
print(f'Therefore as per Cournot\'s condition\n'
      f'\tUnit Price p(q) = a - q = {a_crn_const - q_nash}')
c1_nash = int(input(f'Enter Unit Cost of Production fo Firm1::\n'
                    f'\t+ve Int less than Unit Price {a_crn_const - q_nash} (c1<=p(q)) : '))
c2_nash = int(input(f'Enter Unit Cost of Production fo Firm2::\n'
                    f'\t+ve Int less than Unit Price {a_crn_const - q_nash} (c2<=p(q)) : '))
q_total_qty_arr = np.linspace(0, a_crn_const, a_crn_const + 1)
p_unit_price_arr = a_crn_const - q_total_qty_arr
unconsidered_points_arr = int(np.ceil(a_crn_const * 1.25))


def create_arr_q1_q2_c1(q1_q2=0.):
    p = a_crn_const - q1_q2
    q1 = np.linspace(0, int(q1_q2), int(q1_q2) + 1)
    q2 = q1_q2 - q1
    c1 = np.linspace(0, int(p), 5)[1:4]
    return q1, q2, c1


def calculate_profit(q1, q2, c1):
    profit_1 = q1 * ((a_crn_const - (q1 + q2)) - c1)
    return profit_1


def calculate_q1_for_max_profit(q2, c1):
    return (a_crn_const - q2 - c1) / 2


fig1, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
fig1.suptitle(f'Cournot Competition with \'a\' = {a_crn_const}', fontsize=16)


def draw_default_cournot():
    q_unconsidered_arr = np.linspace(a_crn_const, unconsidered_points_arr,
                                     unconsidered_points_arr - a_crn_const + 1)
    p_unconsidered_arr = np.zeros(len(q_unconsidered_arr))
    ax[0, 0].axvline(0, color='grey', linewidth=1.25)
    ax[0, 0].axhline(0, color='grey', linewidth=1.25)
    ax[0, 0].plot(q_total_qty_arr, p_unit_price_arr, label='Unit Price when q<a', color='red')
    ax[0, 0].plot(q_unconsidered_arr, p_unconsidered_arr, label='Unit Price = 0 when q>=a', color='blue')
    ax[0, 0].set_title('Unit Price, Total Quantity and Cournot\'s Constant (\'a\')')
    ax[0, 0].set(xlabel='Total Quantity Produced (q1+q2) by 2 Firms, F1 & F2', ylabel='Unit Price')
    ax[0, 0].grid()
    ax[0, 0].set_ylim(-5, a_crn_const * 1.1)
    ax[0, 0].set_xlim(-5, unconsidered_points_arr * 1.05)
    ax[0, 0].legend()


def draw_specific_cournot(position=0, q=0., colour='blue'):
    if position == 0:
        ax_pic = ax[0, 1]
    elif position == 1:
        ax_pic = ax[1, 0]
    else:
        ax_pic = ax[1, 1]
    p = a_crn_const - q
    ax_pic.axvline(0, color='grey', linewidth=1.25)
    ax_pic.axhline(0, color='grey', linewidth=1.25)
    ax_pic.plot(q_total_qty_arr, p_unit_price_arr, color=colour, alpha=.5, linewidth=.5, linestyle=':')
    ax_pic.plot([q, q], [0, p], marker='o', markersize=5, linestyle='--', color=colour, linewidth=.5)
    ax_pic.plot([0, q], [p, p], marker='o', markersize=5, linestyle='--', color=colour, linewidth=.5,
                label=f'p(q) = a - q => {p}={a_crn_const}-{q}\nq=q1+q2 = {q}\n')
    ax_pic.set_title(f'Unit Price={p}, Quantity(q1+q2)={q}')
    ax_pic.set(xlabel='Quantity', ylabel='Unit Price')
    ax_pic.grid()
    ax_pic.set_ylim(-5, a_crn_const * 1.1)
    ax_pic.set_xlim(-5, unconsidered_points_arr * 1.05)
    ax_pic.legend()


def draw_profit_vs_quantities(q1, q2, c, q1_q2=0., color='blue'):
    fig2 = plt.figure(figsize=(12, 5))
    fig2.suptitle(f'Profit vs. Quantities for a = {a_crn_const}, quantity(q1+q2)={q1_q2}\n'
                  f'Unit Price = a - (q1+q2) = {a_crn_const - q1_q2}', fontsize=12)
    for position in range(1, 4):
        ax_pic = fig2.add_subplot(1, 3, position, projection='3d')
        q1_mesh, q2_mesh = np.meshgrid(q1, q2)
        p_mesh = calculate_profit(q1_mesh, q2_mesh, c[position - 1])
        ax_pic.plot_surface(q1_mesh, q2_mesh, p_mesh, color=color, alpha=0.5, linewidth=.5)

        q1_for_max = calculate_q1_for_max_profit(q2_arr, c1_cost_of_prod_arr[position - 1])
        max_profit_ln = calculate_profit(q1_for_max, q2_arr, c1_cost_of_prod_arr[position - 1])
        ax_pic.plot3D(q1_for_max, q2, max_profit_ln, color=color, linewidth=1.5)

        ax_pic.set_xlim(0, len(q1))
        ax_pic.set_ylim(0, len(q2))
        ax_pic.set_xlabel('q1 -->')
        ax_pic.set_ylabel('q2 -->')
        ax_pic.set_zlabel('Profit -->')
        ax_pic.set_title(f'Cost of Production c1 ={c[position - 1]}')
    fig2.tight_layout()
    fig2.savefig(f'./plots/02Cournot_ProfitVsQuantity_{q1_q2}.png')


draw_default_cournot()
colors = ['green', 'orange', 'purple']
fraction_arr = np.linspace(0, 1, 5)[1:4]

for i in np.arange(3):
    q1_q2_sum = a_crn_const * fraction_arr[i]
    draw_specific_cournot(i, q1_q2_sum, colors[i])
    q1_arr, q2_arr, c1_cost_of_prod_arr = create_arr_q1_q2_c1(q1_q2_sum)
    draw_profit_vs_quantities(q1_arr, q2_arr, c1_cost_of_prod_arr, q1_q2_sum, colors[i])

fig3 = plt.figure(figsize=plt.figaspect(0.7))
fig3.suptitle(f'Nash Equilibrium for a = {a_crn_const}, quantity(q1+q2)={q_nash}\n'
              f'Unit Price = a - q = {a_crn_const - q_nash}', fontsize=12)
ax = fig3.add_subplot(1, 1, 1, projection='3d')

q1_nash = np.linspace(0, int(q_nash), int(q_nash) + 1)
q2_nash = q_nash - q1_nash
q1_mesh, q2_mesh = np.meshgrid(q1_nash, q2_nash)

p1_mesh = calculate_profit(q1_mesh, q2_mesh, c1_nash)
ax.plot_surface(q1_mesh, q2_mesh, p1_mesh, color='red', alpha=0.2, linewidth=.5, label=f'Firm-1: c1={c1_nash}')
q1_for_max = calculate_q1_for_max_profit(q2_nash, c1_nash)
max_profit_ln = calculate_profit(q1_for_max, q2_nash, c1_nash)
ax.plot3D(q1_for_max, q2_nash, max_profit_ln, color='red', linewidth=1.5, label='Firm-1 Max Profit')

p2_mesh = calculate_profit(q1_mesh, q2_mesh, c2_nash)
ax.plot_surface(q2_mesh, q1_mesh, p2_mesh, color='green', alpha=0.2, linewidth=.5, label=f'Firm-2: c2={c2_nash}')
q2_for_max = calculate_q1_for_max_profit(q1_nash, c2_nash)
max_profit_ln = calculate_profit(q2_for_max, q1_nash, c2_nash)
ax.plot3D(q1_nash, q2_for_max, max_profit_ln, color='green', linewidth=1.5, label='Firm-2 Max Profit')

game = nash.Game(p1_mesh, p2_mesh)
nash_equilibrium = game.support_enumeration()
# next_nash = next(nash_equilibrium)
print(f'nash_equilibrium({type(nash_equilibrium)})\n{nash_equilibrium}')

ax.legend(loc='upper right')
ax.set_xlim(0, len(q1_nash))
ax.set_ylim(0, len(q2_nash))
ax.set_xlabel('q1 -->')
ax.set_ylabel('q2 -->')
ax.set_zlabel('Profit -->')

fig1.savefig(f'./plots/02Cournot_Competition_a_{a_crn_const}.png')
fig3.savefig(f'./plots/02Cournot_NashEq_a_{a_crn_const}_q_{q_nash}.png')
plt.show()
