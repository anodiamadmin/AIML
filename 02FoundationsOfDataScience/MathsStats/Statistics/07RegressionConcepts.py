import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use('seaborn-v0_8')
x_vals = np.linspace(-3, 6, 7)
y_scatter = np.array([0.85, -1.88, 6.43, 11.11, 4.96, 11.5, 17])
x_trend = np.linspace(-6, 9, 11)

figProb, axProb = plt.subplots(figsize=(5, 5))
axProb.scatter(x_vals, y_scatter, color='#ff3355', alpha=0.8)
legend_string = '     Points:'
for index in range(len(x_vals)):
    legend_string += f'\n{index + 1}. {x_vals[index], y_scatter[index]}'
figProb.text(.6, .2, bbox=dict(facecolor='yellow', alpha=0.5), s=legend_string)
axProb.set_title("Draw trend line through the points & predict y for x=8")
axProb.set_xlim([-10, 30])
axProb.set_ylim([-10, 30])
axProb.set_xlabel("x ->")
axProb.set_ylabel("y ->")
axProb.axhline(0, color='k', alpha=0.5, linewidth=.5)   # x axis
axProb.axvline(0, color='k', alpha=0.5, linewidth=.5)   # y axis
plt.grid(True)
plt.tight_layout()
plt.show()
figProb.savefig('./plots/07RegressionConcepts_Problem.png')

slope_cache = np.array([])
intercept_cache = np.array([])
continue_game = 'y'
print(f'Try to Draw TREND the line:')
init_slope, init_intercept, init_err, try_num = 0, -20, -1, 1
while (continue_game != 'N' and continue_game != 'n' and continue_game != 'No'
       and continue_game != 'no' and continue_game != 'NO' and try_num < 5):
    intercept = float(input("Enter Intercept: (Hint: Float between: -5 and 15): "))
    slope = float(input("Enter Slope: (Hint: Float between: -5 and 5): "))
    figTry, axTry = plt.subplots(figsize=(5, 5))
    x_vals = np.linspace(-3, 6, 7)
    y_scatter = np.array([0.85, -1.88, 6.43, 11.11, 4.96, 11.5, 17])
    y_vals = slope * x_vals + intercept
    y_trend = slope * x_trend + intercept
    y_8 = slope * 8 + intercept
    y_init_trend = init_slope * x_trend + init_intercept
    data = np.array([x_vals, y_scatter]).T
    axTry.scatter(x_vals, y_scatter, color='#ff3355', alpha=0.8)
    axTry.plot(x_trend, y_trend, color='#3355ff', alpha=0.8)
    axTry.plot(x_trend, y_init_trend, color='#888888', alpha=0.8, linewidth=1, linestyle=':')
    axTry.scatter(8, y_8, color='green', marker='*', s=100)
    axTry.set_title("Your trend line through the 7 points")
    axTry.set_xlim([-10, 30])
    axTry.set_ylim([-10, 30])
    axTry.set_xlabel("x ->")
    axTry.set_ylabel("y ->")
    axTry.axhline(0, color='k', alpha=0.5, linewidth=.5)  # x-axis
    axTry.axvline(0, color='k', alpha=0.5, linewidth=.5)  # y-axis
    err_arr = np.zeros(len(x_vals))
    sqr_err = 0
    legend_error_string = '     Errors:'
    for index in range(len(x_vals)):
        axTry.plot([x_vals[index], x_vals[index]], [y_vals[index], y_scatter[index]],
                   color='#aa3355', alpha=0.8, linewidth=.5, linestyle=':')
        sqr_err += ((y_vals[index] - y_scatter[index]) ** 2) / len(x_vals)
        legend_error_string += (f'\n{index + 1}. {x_vals[index], y_scatter[index]}: '
                                f'{np.round(((y_vals[index] - y_scatter[index]) ** 2) / len(x_vals), 2)}')
    legend_error_string += f'\nStdErr:({intercept}+{slope}x)={np.round(sqr_err, 2)}'
    if init_err >= 0:
        legend_error_string += f'\nPrvErr:({init_intercept}+{init_slope}x)={np.round(init_err, 2)}'
    figTry.text(.6, .2, bbox=dict(facecolor='yellow', alpha=0.5), s=legend_error_string)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    init_slope, init_intercept, init_err, try_num = slope, intercept, sqr_err, try_num + 1
    slope_cache = np.append(slope_cache, init_slope)
    intercept_cache = np.append(intercept_cache, init_intercept)
    continue_game = input(f"Try AGAIN? Y/N: ")
print(f'THANK YOU FOR PLAYING!\nHere is the solution: intercept = 4.51, slope = 1.76')
print(f'Your Choices:\nintercept_cache={intercept_cache}, slope_cache={slope_cache}')

slope, intercept, r, p, sqr_err = stats.linregress(x_vals, y_scatter)
y_hat = slope * x_vals + intercept
y_trend = slope * x_trend + intercept
y_8 = slope * 8 + intercept
figSol, axSol = plt.subplots(figsize=(5, 5))
axSol.scatter(x_vals, y_scatter, color='#ff3355', alpha=0.8)
axSol.plot(x_trend, y_trend, color='#33ff55', alpha=0.8)
axSol.set_title("Solution: Trend line with Minimum Standard Error")
axSol.set_xlim([-10, 30])
axSol.set_ylim([-10, 30])
axSol.set_xlabel("x ->")
axSol.set_ylabel("y ->")
axSol.axhline(0, color='k', alpha=0.5, linewidth=.5)   # x axis
axSol.axvline(0, color='k', alpha=0.5, linewidth=.5)   # y axis
axSol.scatter(8, y_8, color='green', marker='*', s=100)
err_arr = np.zeros(len(x_vals))
sqr_err = 0
legend_error_string = '     Errors:'
for index in range(len(x_vals)):
    axSol.plot([x_vals[index], x_vals[index]], [y_hat[index], y_scatter[index]],
               color='#aa3355', alpha=0.8, linewidth=.5, linestyle=':')
    sqr_err += ((y_hat[index] - y_scatter[index]) ** 2) / len(x_vals)
    legend_error_string += (f'\n{index + 1}. {x_vals[index], y_scatter[index]}: '
                            f'{np.round(((y_hat[index] - y_scatter[index]) ** 2) / len(x_vals), 2)}')
legend_error_string += (f'\nStdErr:({np.round(intercept, 2)}+{np.round(slope, 2)}x)='
                        f'{np.round(sqr_err, 2)}')
figSol.text(.6, .2, bbox=dict(facecolor='yellow', alpha=0.5), s=legend_error_string)
plt.grid(True)
plt.tight_layout()
plt.show()
figProb.savefig('./plots/07RegressionConcepts_Solution.png')

intercept_range = np.linspace(-20, 35, 101)
slope_range = np.linspace(-10, 13.5, 101)
intercept_range, slope_range = np.meshgrid(intercept_range, slope_range)


def calc_sqr_err(intercept, slope):
    sqr_err = 0
    for index in range(len(x_vals)):
        sqr_err += ((intercept + slope * x_vals[index]) - y_scatter[index]) ** 2
    return sqr_err / len(x_vals)


v_calc_sqr_err = np.vectorize(calc_sqr_err)
v_sqr_err = v_calc_sqr_err(intercept_range, slope_range)

fig = plt.figure(figsize=plt.figaspect(0.4))
elevation, roll, azimuth = 30, 0, 30
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax0.plot_surface(intercept_range, slope_range, v_sqr_err, color='purple', alpha=0.2, linewidth=.5)
ax0.plot_surface(intercept_range, slope_range, np.zeros((len(intercept_range), len(slope_range))),
                 color='#8899cc', alpha=0.2, linewidth=.3)
ax0.scatter(4.51, 1.76, calc_sqr_err(4.51, 1.76), color='blue', alpha=0.8,
            marker='*', label=f'Minima: Err={np.round(calc_sqr_err(4.51, 1.76), 2)}'
                              f'\n Pt = (4.51,1.76')
ax0.view_init(elevation, azimuth, roll)
ax0.set_title("SLR - Error Minimization - Cost Optimization")
ax0.set_xlabel("<-- Intercept")
ax0.set_ylabel("<-- Slope")
ax0.set_zlabel("Squared Error -->")
ax0.legend(loc='upper right')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
last_pt = np.full(3, -99)
for index in range(len(intercept_cache)):
    ax1.scatter(intercept_cache[index], slope_cache[index],
                calc_sqr_err(intercept_cache[index], slope_cache[index]),
                color='#0099dd', alpha=0.8, marker='*', s=10,
                label=f'Pt{index+1}. ({intercept_cache[index]}, {slope_cache[index]}) '
                      f'Err={np.round(calc_sqr_err(intercept_cache[index], slope_cache[index]), 2)}')
    ax1.scatter(intercept_cache[index], slope_cache[index], 0,
                color='orange', alpha=0.8, marker='o', s=10)
    ax1.plot([intercept_cache[index], intercept_cache[index]], [slope_cache[index], slope_cache[index]],
             [0, calc_sqr_err(intercept_cache[index], slope_cache[index])],
             color='orange', alpha=0.8, linewidth=1, linestyle='-')
    if last_pt[2] != -99:
        ax1.plot([intercept_cache[index], last_pt[0]], [slope_cache[index], last_pt[1]],
                 [calc_sqr_err(intercept_cache[index], slope_cache[index]), last_pt[2]],
                 color='green', alpha=0.8, linewidth=1, linestyle='-')
    last_pt = [intercept_cache[index], slope_cache[index], calc_sqr_err(intercept_cache[index], slope_cache[index])]
ax1.plot_surface(intercept_range, slope_range, v_sqr_err, color='purple', alpha=0.2, linewidth=.5)
ax1.plot_surface(intercept_range, slope_range, np.zeros((len(intercept_range), len(slope_range))),
                 color='#8899cc', alpha=0.2, linewidth=.3)
ax1.view_init(elevation, azimuth, roll)
ax1.set_title("Your Path of Gradiant Descent")
ax1.set_xlabel("<-- Intercept")
ax1.set_ylabel("<-- Slope")
ax1.set_zlabel("Squared Error -->")
ax1.legend(loc='upper right')
plt.tight_layout()
plt.savefig('./plots/07SLR_CostOptimization.png')
plt.show()
