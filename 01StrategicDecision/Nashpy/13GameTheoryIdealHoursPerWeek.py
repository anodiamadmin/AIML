import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Data Preparation
np.random.seed(41)
total_employees = 2000
mean_timesheet = 40*60
std_timesheet = 4*60

employee_timesheet = np.random.normal(mean_timesheet, std_timesheet, total_employees)
# print(f'employee_timesheet = {employee_timesheet}')
psn_arr = np.random.poisson(total_employees, total_employees)
# print(f'psn_arr = {psn_arr}')
employee_satisfaction = ((mean_timesheet + 100 * std_timesheet - employee_timesheet) * psn_arr / 10000000 - 4) * 4.1
# print(f'employee_satisfaction = {employee_satisfaction}')
median_satisfaction = np.percentile(employee_satisfaction, 50)

q_33p = np.percentile(employee_timesheet, 100/3)
q_67p = np.percentile(employee_timesheet, 2*100/3)
# print(f'q_33p = {q_33p}')
# print(f'q_67p = {q_67p}')

rnd_psn_arr = np.random.random(total_employees) * 700 + psn_arr
# print(f'nml_arr = {rnd_psn_arr}')
employee_performance = ((mean_timesheet + 100 * std_timesheet - employee_timesheet) * rnd_psn_arr / 10000000 - 4) * 1.75
# print(f'employee_performance = {employee_performance}')
q_33prf = np.percentile(employee_performance, 100/3)
q_67prf = np.percentile(employee_performance, 2*100/3)
mean_performance = np.mean(employee_performance)
# print(f'mean_performance = {mean_performance}')
employee_number = np.arange(1, total_employees + 1)
employee_data = np.array([np.round(employee_number), np.round(employee_timesheet),
                          np.round(employee_satisfaction, 1),
                          np.round(employee_performance, 1)]).T
# print(employee_data.shape)
# print(employee_data)
##############################
np.savetxt('./data/13GameTheoryIdealHoursPerWeek.csv', employee_data, delimiter=',', fmt='%10.1f')


# Data Visualization:
plt.style.use('ggplot')

fig, ax = plt.subplots(2, 2, sharex=False, sharey=False)
fig.set_figheight(6)
fig.set_figwidth(10)

ax[0][0].hist(employee_timesheet, bins=np.arange(mean_timesheet - 4 * std_timesheet,
                                                 mean_timesheet + 4 * std_timesheet, 15),
              edgecolor='black', color='pink', linewidth=.3)
ax[0][0].axvline(q_33p, alpha=.5, color='green', linewidth=1.5, linestyle='--',
                 label=f'33% employees work\nup to {np.round(q_33p/60, 1)}hrs')
ax[0][0].axvline(q_67p, alpha=.5, color='red', linewidth=1.5, linestyle='--',
                 label=f'67% employees work\nup to {np.round(q_67p/60, 1)}hrs')
ax[0][0].set_title("Hours Worked Per Week")
ax[0][0].set_xlabel("Hours ->")
ax[0][0].set_ylabel("Number of Employees ->")
ax[0][0].set_xlim(mean_timesheet - 4 * std_timesheet, mean_timesheet + 4 * std_timesheet)
ax[0][0].set_ylim(-5, 75)
ax[0][0].set_xticks(ticks=range(mean_timesheet - 4 * std_timesheet,
                                mean_timesheet + 4 * std_timesheet, std_timesheet))
ax[0][0].set_xticklabels(labels=range(int((mean_timesheet - 4 * std_timesheet) / 60),
                                      int((mean_timesheet + 4 * std_timesheet) / 60),
                                      int(std_timesheet / 60)),
                         rotation=90)
ax[0][0].legend(loc="upper left", fontsize=8)

sns.regplot(x=employee_timesheet, y=employee_satisfaction, ax=ax[0][1], marker='o', color='#333355',
            scatter_kws={'s': 1}, line_kws={'color': '#8888aa', 'label': 'Trend Line'})

ax[0][1].axhline(median_satisfaction, alpha=1, color='#8888aa', linewidth=1.5, linestyle='-.',
                 label=f'Median={np.round(median_satisfaction, 1)}')
ax[0][1].set_xticks(ticks=range(mean_timesheet - 4 * std_timesheet,
                                mean_timesheet + 4 * std_timesheet, std_timesheet))
ax[0][1].set_xticklabels(labels=range(int((mean_timesheet - 4 * std_timesheet) / 60),
                                      int((mean_timesheet + 4 * std_timesheet) / 60),
                                      int(std_timesheet / 60)),
                         rotation=90)
ax[0][1].set_title("Hours Worked Vs Employee Satisfaction")
ax[0][1].set_xlabel("Hours ->")
ax[0][1].set_ylabel("Satisfaction Rating ->")
ax[0][1].axvline(q_33p, alpha=.5, color='green', linewidth=1.5, linestyle='--')
ax[0][1].axvline(q_67p, alpha=.5, color='red', linewidth=1.5, linestyle='--')
ax[0][1].legend(loc="lower left", fontsize=8)

sns.regplot(x=employee_timesheet, y=employee_performance, ax=ax[1][0], marker='o', color='#333355',
            scatter_kws={'s': 1}, line_kws={'color': '#8888aa', 'label': 'Trend Line'})
ax[1][0].set_title("Hours Worked Vs Performance Rating")
ax[1][0].set_xlabel("Hours ->")
ax[1][0].set_ylabel("Performance Rating ->")
ax[1][0].set_xlim(mean_timesheet - 4 * std_timesheet, mean_timesheet + 4 * std_timesheet)
ax[1][0].set_xticks(ticks=range(mean_timesheet - 4 * std_timesheet,
                                mean_timesheet + 4 * std_timesheet, std_timesheet))
ax[1][0].set_xticklabels(labels=range(int((mean_timesheet - 4 * std_timesheet) / 60),
                                      int((mean_timesheet + 4 * std_timesheet) / 60),
                                      int(std_timesheet / 60)),
                         rotation=90)
ax[1][0].axvline(q_33p, alpha=.5, color='green', linewidth=1.5, linestyle='--')
ax[1][0].axvline(q_67p, alpha=.5, color='red', linewidth=1.5, linestyle='--')
ax[1][0].axhline(mean_performance, alpha=1, color='#8888aa', linewidth=1.5, linestyle='-.',
                 label=f'Mean={np.round(mean_performance, 1)}')
ax[1][0].legend(loc="lower left", fontsize=8)
###########################


# Game Theory
# happy_array = employee_data[np.where(employee_data[:, 2] > median_satisfaction)]
# unhappy_array = employee_data[np.where(employee_data[:, 2] <= median_satisfaction)]
# # print(f'happy_array = {happy_array}')
# # print(f'unhappy_array = {unhappy_array}')
# # print(f'happy_array.shape = {happy_array.shape}')
# # print(f'unhappy_array.shape = {unhappy_array.shape}')
#
# q_happ_33p = np.percentile(happy_array[:, 1], 100/3)
# q_happ_67p = np.percentile(happy_array[:, 1], 2*100/3)
# happy_low_hrs = happy_array[np.where((happy_array[:, 1] < q_happ_33p))]
# happy_med_hrs = happy_array[np.where((happy_array[:, 1] >= q_happ_33p) & (happy_array[:, 1] < q_happ_67p))]
# happy_hi_hrs = happy_array[np.where((happy_array[:, 1] >= q_happ_67p))]
# print(f'happy_low_hrs mean Perf = {np.mean(happy_low_hrs.T[3])}')
# print(f'happy_mid_hrs mean Perf = {np.mean(happy_med_hrs.T[3])}')
# print(f'happy_hi_hrs mean Perf = {np.mean(happy_hi_hrs.T[3])}')
#
# q_unhapp_33p = np.percentile(unhappy_array[:, 1], 100/3)
# q_unhapp_67p = np.percentile(unhappy_array[:, 1], 2*100/3)
# unhappy_low_hrs = unhappy_array[np.where((unhappy_array[:, 1] < q_unhapp_33p))]
# unhappy_med_hrs = unhappy_array[np.where((unhappy_array[:, 1] >= q_unhapp_33p) & (unhappy_array[:, 1] < q_unhapp_67p))]
# unhappy_hi_hrs = unhappy_array[np.where((unhappy_array[:, 1] >= q_unhapp_67p))]
# print(f'UNhappy_low_hrs mean Perf = {np.mean(unhappy_low_hrs.T[3])}')
# print(f'UNhappy_mid_hrs mean Perf = {np.mean(unhappy_med_hrs.T[3])}')
# print(f'UNhappy_hi_hrs mean Perf = {np.mean(unhappy_hi_hrs.T[3])}')
#
# q_happ_lo_33prf = np.percentile(happy_low_hrs[:, 3], 100/3)
# q_happ_lo_67prf = np.percentile(happy_low_hrs[:, 3], 2*100/3)
# happy_low_hrs_hi_perf = np.round(np.mean(happy_low_hrs[np.where((happy_low_hrs[:, 3] >= q_happ_lo_67prf))].T[3]), 1)
# happy_low_hrs_mid_perf = np.round(np.mean(happy_low_hrs[np.where((happy_low_hrs[:, 3] >= q_happ_lo_33prf) & (happy_low_hrs[:, 3] < q_happ_lo_67prf))].T[3]), 1)
# happy_low_hrs_low_perf = np.round(np.mean(happy_low_hrs[np.where((happy_low_hrs[:, 3] < q_happ_lo_33prf))].T[3]), 1)
# print(f'happy_low_hrs_hi_perf = {happy_low_hrs_hi_perf}')
# print(f'happy_low_hrs_mid_perf = {happy_low_hrs_mid_perf}')
# print(f'happy_low_hrs_low_perf = {happy_low_hrs_low_perf}')
#
# q_happ_med_33prf = np.percentile(happy_med_hrs[:, 3], 100/3)
# q_happ_med_67prf = np.percentile(happy_med_hrs[:, 3], 2*100/3)
# happy_med_hrs_hi_perf = np.round(np.mean(happy_med_hrs[np.where((happy_med_hrs[:, 3] >= q_happ_med_67prf))].T[3]), 1)
# happy_med_hrs_mid_perf = np.round(np.mean(happy_med_hrs[np.where((happy_med_hrs[:, 3] >= q_happ_med_33prf) & (happy_med_hrs[:, 3] < q_happ_med_67prf))].T[3]), 1)
# happy_med_hrs_low_perf = np.round(np.mean(happy_med_hrs[np.where((happy_med_hrs[:, 3] < q_happ_med_33prf))].T[3]), 1)
# print(f'happy_med_hrs_hi_perf = {happy_med_hrs_hi_perf}')
# print(f'happy_med_hrs_mid_perf = {happy_med_hrs_mid_perf}')
# print(f'happy_med_hrs_low_perf = {happy_med_hrs_low_perf}')
#
# q_happ_hi_33prf = np.percentile(happy_hi_hrs[:, 3], 100/3)
# q_happ_hi_67prf = np.percentile(happy_hi_hrs[:, 3], 2*100/3)
# happy_hi_hrs_hi_perf = np.round(np.mean(happy_hi_hrs[np.where((happy_hi_hrs[:, 3] >= q_happ_hi_67prf))].T[3]), 1)
# happy_hi_hrs_mid_perf = np.round(np.mean(happy_hi_hrs[np.where((happy_hi_hrs[:, 3] >= q_happ_hi_33prf) & (happy_hi_hrs[:, 3] < q_happ_hi_67prf))].T[3]), 1)
# happy_hi_hrs_low_perf = np.round(np.mean(happy_hi_hrs[np.where((happy_hi_hrs[:, 3] < q_happ_hi_33prf))].T[3]), 1)
# print(f'happy_hi_hrs_hi_perf = {happy_hi_hrs_hi_perf}')
# print(f'happy_hi_hrs_mid_perf = {happy_hi_hrs_mid_perf}')
# print(f'happy_hi_hrs_low_perf = {happy_hi_hrs_low_perf}')
#
# q_unhapp_lo_33prf = np.percentile(unhappy_low_hrs[:, 3], 100/3)
# q_unhapp_lo_67prf = np.percentile(unhappy_low_hrs[:, 3], 2*100/3)
# unhappy_low_hrs_hi_perf = np.round(np.mean(unhappy_low_hrs[np.where((unhappy_low_hrs[:, 3] >= q_unhapp_lo_67prf))].T[3]), 1)
# unhappy_low_hrs_mid_perf = np.round(np.mean(unhappy_low_hrs[np.where((unhappy_low_hrs[:, 3] >= q_unhapp_lo_33prf) & (unhappy_low_hrs[:, 3] < q_unhapp_lo_67prf))].T[3]), 1)
# unhappy_low_hrs_low_perf = np.round(np.mean(unhappy_low_hrs[np.where((unhappy_low_hrs[:, 3] < q_unhapp_lo_33prf))].T[3]), 1)
# print(f'unhappy_low_hrs_hi_perf = {unhappy_low_hrs_hi_perf}')
# print(f'unhappy_low_hrs_mid_perf = {unhappy_low_hrs_mid_perf}')
# print(f'unhappy_low_hrs_low_perf = {unhappy_low_hrs_low_perf}')
#
# q_unhapp_med_33prf = np.percentile(unhappy_med_hrs[:, 3], 100/3)
# q_unhapp_med_67prf = np.percentile(unhappy_med_hrs[:, 3], 2*100/3)
# unhappy_med_hrs_hi_perf = np.round(np.mean(unhappy_med_hrs[np.where((unhappy_med_hrs[:, 3] >= q_unhapp_med_67prf))].T[3]), 1)
# unhappy_med_hrs_mid_perf = np.round(np.mean(unhappy_med_hrs[np.where((unhappy_med_hrs[:, 3] >= q_unhapp_med_33prf) & (unhappy_med_hrs[:, 3] < q_unhapp_med_67prf))].T[3]), 1)
# unhappy_med_hrs_low_perf = np.round(np.mean(unhappy_med_hrs[np.where((unhappy_med_hrs[:, 3] < q_unhapp_med_33prf))].T[3]), 1)
# print(f'unhappy_med_hrs_hi_perf = {unhappy_med_hrs_hi_perf}')
# print(f'unhappy_med_hrs_mid_perf = {unhappy_med_hrs_mid_perf}')
# print(f'unhappy_med_hrs_low_perf = {unhappy_med_hrs_low_perf}')
#
# q_unhapp_hi_33prf = np.percentile(unhappy_hi_hrs[:, 3], 100/3)
# q_unhapp_hi_67prf = np.percentile(unhappy_hi_hrs[:, 3], 2*100/3)
# unhappy_hi_hrs_hi_perf = np.round(np.mean(unhappy_hi_hrs[np.where((unhappy_hi_hrs[:, 3] >= q_unhapp_hi_67prf))].T[3]), 1)
# unhappy_hi_hrs_mid_perf = np.round(np.mean(unhappy_hi_hrs[np.where((unhappy_hi_hrs[:, 3] >= q_unhapp_hi_33prf) & (unhappy_hi_hrs[:, 3] < q_unhapp_hi_67prf))].T[3]), 1)
# unhappy_hi_hrs_low_perf = np.round(np.mean(unhappy_hi_hrs[np.where((unhappy_hi_hrs[:, 3] < q_unhapp_hi_33prf))].T[3]), 1)
# print(f'unhappy_hi_hrs_hi_perf = {unhappy_hi_hrs_hi_perf}')
# print(f'unhappy_hi_hrs_mid_perf = {unhappy_hi_hrs_mid_perf}')
# print(f'unhappy_hi_hrs_low_perf = {unhappy_hi_hrs_low_perf}')

mdn_ts = np.median(employee_timesheet)
# print(f'mdn_ts = {mdn_ts}')
whol_arr = employee_data[np.where(employee_timesheet >= mdn_ts)]
wlb_arr = employee_data[np.where(employee_timesheet < mdn_ts)]
# print(f'whol_arr.shape = {whol_arr.shape}')
# print(f'wlb_arr.shape = {wlb_arr.shape}')
# print(f'whol_arr mean Perf = {np.mean(whol_arr.T[3])}')
# print(f'wlb_arr mean Perf = {np.mean(wlb_arr.T[3])}')
# print(f'whol_arr = {whol_arr}')
# print(f'wlb_arr = {wlb_arr}')

q33_whol_hap = np.percentile(whol_arr[:, 2], 100/3)
q67_whol_hap = np.percentile(whol_arr[:, 2], 2*100/3)
# print(f'q33_whol_hap = {q33_whol_hap}')
# print(f'q67_whol_hap = {q67_whol_hap}')
whol_low_happ_arr = whol_arr[np.where((whol_arr[:, 2] < q33_whol_hap))]
whol_med_happ_arr = whol_arr[np.where((whol_arr[:, 2] >= q33_whol_hap) & (whol_arr[:, 2] < q67_whol_hap))]
whol_hi_happ_arr = whol_arr[np.where((whol_arr[:, 2] >= q67_whol_hap))]
# print(f'whol_low_happ_arr mean Perf = {np.mean(whol_low_happ_arr.T[3])}')
# print(f'whol_med_happ_arr mean Perf = {np.mean(whol_med_happ_arr.T[3])}')
# print(f'whol_hi_happ_arr mean Perf = {np.mean(whol_hi_happ_arr.T[3])}')
q33_whol_low_prf = np.percentile(whol_low_happ_arr[:, 3], 100/3)
q67_whol_low_prf = np.percentile(whol_low_happ_arr[:, 3], 2*100/3)
whol_low_happ_low_perf = np.round(np.mean(whol_low_happ_arr[np.where((whol_low_happ_arr[:, 3] < q33_whol_low_prf))].T[3]), 1)
whol_low_happ_mid_perf = np.round(np.mean(whol_low_happ_arr[np.where((whol_low_happ_arr[:, 3] >= q33_whol_low_prf) & (whol_low_happ_arr[:, 3] < q67_whol_low_prf))].T[3]), 1)
whol_low_happ_hi_perf = np.round(np.mean(whol_low_happ_arr[np.where((whol_low_happ_arr[:, 3] >= q67_whol_low_prf))].T[3]), 1)
print(f'whol_low_happ_low_perf = {whol_low_happ_low_perf}')
print(f'whol_low_happ_mid_perf = {whol_low_happ_mid_perf}')
print(f'whol_low_happ_hi_perf = {whol_low_happ_hi_perf}')

q33_whol_med_prf = np.percentile(whol_med_happ_arr[:, 3], 100/3)
q67_whol_med_prf = np.percentile(whol_med_happ_arr[:, 3], 2*100/3)
whol_med_happ_low_perf = np.round(np.mean(whol_med_happ_arr[np.where((whol_med_happ_arr[:, 3] < q33_whol_med_prf))].T[3]), 1)
whol_med_happ_mid_perf = np.round(np.mean(whol_med_happ_arr[np.where((whol_med_happ_arr[:, 3] >= q33_whol_med_prf) & (whol_med_happ_arr[:, 3] < q67_whol_med_prf))].T[3]), 1)
whol_med_happ_hi_perf = np.round(np.mean(whol_med_happ_arr[np.where((whol_med_happ_arr[:, 3] >= q67_whol_med_prf))].T[3]), 1)
print(f'whol_med_happ_low_perf = {whol_med_happ_low_perf}')
print(f'whol_med_happ_mid_perf = {whol_med_happ_mid_perf}')
print(f'whol_med_happ_hi_perf = {whol_med_happ_hi_perf}')

q33_whol_hi_prf = np.percentile(whol_hi_happ_arr[:, 3], 100/3)
q67_whol_hi_prf = np.percentile(whol_hi_happ_arr[:, 3], 2*100/3)
whol_hi_happ_low_perf = np.round(np.mean(whol_hi_happ_arr[np.where((whol_hi_happ_arr[:, 3] < q33_whol_hi_prf))].T[3]), 1)
whol_hi_happ_mid_perf = np.round(np.mean(whol_hi_happ_arr[np.where((whol_hi_happ_arr[:, 3] >= q33_whol_hi_prf) & (whol_hi_happ_arr[:, 3] < q67_whol_hi_prf))].T[3]), 1)
whol_hi_happ_hi_perf = np.round(np.mean(whol_hi_happ_arr[np.where((whol_hi_happ_arr[:, 3] >= q67_whol_hi_prf))].T[3]), 1)
print(f'whol_hi_happ_low_perf = {whol_hi_happ_low_perf}')
print(f'whol_hi_happ_mid_perf = {whol_hi_happ_mid_perf}')
print(f'whol_hi_happ_hi_perf = {whol_hi_happ_hi_perf}')

# print(f'q33_whol_low_prf = {q33_whol_low_prf}')
# print(f'q67_whol_low_prf = {q67_whol_low_prf}')
# print(f'q33_whol_med_prf = {q33_whol_med_prf}')
# print(f'q67_whol_med_prf = {q67_whol_med_prf}')
# print(f'q33_whol_hi_prf = {q33_whol_hi_prf}')
# print(f'q67_whol_hi_prf = {q67_whol_hi_prf}')

q33_wlb_hap = np.percentile(wlb_arr[:, 2], 100/3)
q67_wlb_hap = np.percentile(wlb_arr[:, 2], 2*100/3)
# print(f'q33_wlb_hap = {q33_wlb_hap}')
# print(f'q67_wlb_hap = {q67_wlb_hap}')
wlb_low_happ_arr = wlb_arr[np.where((wlb_arr[:, 2] < q33_wlb_hap))]
wlb_med_happ_arr = wlb_arr[np.where((wlb_arr[:, 2] >= q33_wlb_hap) & (wlb_arr[:, 2] < q67_wlb_hap))]
wlb_hi_happ_arr = wlb_arr[np.where((wlb_arr[:, 2] >= q67_wlb_hap))]
# print(f'wlb_low_happ_arr mean Perf = {np.mean(wlb_low_happ_arr.T[3])}')
# print(f'wlb_med_happ_arr mean Perf = {np.mean(wlb_med_happ_arr.T[3])}')
# print(f'wlb_hi_happ_arr mean Perf = {np.mean(wlb_hi_happ_arr.T[3])}')
q33_wlb_low_prf = np.percentile(wlb_low_happ_arr[:, 3], 100/3)
q67_wlb_low_prf = np.percentile(wlb_low_happ_arr[:, 3], 2*100/3)
wlb_low_happ_low_perf = np.round(np.mean(wlb_low_happ_arr[np.where((wlb_low_happ_arr[:, 3] < q33_wlb_low_prf))].T[3]), 1)
wlb_low_happ_mid_perf = np.round(np.mean(wlb_low_happ_arr[np.where((wlb_low_happ_arr[:, 3] >= q33_wlb_low_prf) & (wlb_low_happ_arr[:, 3] < q67_wlb_low_prf))].T[3]), 1)
wlb_low_happ_hi_perf = np.round(np.mean(wlb_low_happ_arr[np.where((wlb_low_happ_arr[:, 3] >= q67_wlb_low_prf))].T[3]), 1)
print(f'wlb_low_happ_low_perf = {wlb_low_happ_low_perf}')
print(f'wlb_low_happ_mid_perf = {wlb_low_happ_mid_perf}')
print(f'wlb_low_happ_hi_perf = {wlb_low_happ_hi_perf}')

q33_wlb_med_prf = np.percentile(wlb_med_happ_arr[:, 3], 100/3)
q67_wlb_med_prf = np.percentile(wlb_med_happ_arr[:, 3], 2*100/3)
wlb_med_happ_low_perf = np.round(np.mean(wlb_med_happ_arr[np.where((wlb_med_happ_arr[:, 3] < q33_wlb_med_prf))].T[3]), 1)
wlb_med_happ_med_perf = np.round(np.mean(wlb_med_happ_arr[np.where((wlb_med_happ_arr[:, 3] >= q33_wlb_med_prf) & (wlb_med_happ_arr[:, 3] < q67_wlb_med_prf))].T[3]), 1)
wlb_med_happ_hi_perf = np.round(np.mean(wlb_med_happ_arr[np.where((wlb_med_happ_arr[:, 3] >= q67_wlb_med_prf))].T[3]), 1)
print(f'wlb_med_happ_low_perf = {wlb_med_happ_low_perf}')
print(f'wlb_med_happ_mid_perf = {wlb_med_happ_med_perf}')
print(f'wlb_med_happ_hi_perf = {wlb_med_happ_hi_perf}')

q33_wlb_hi_prf = np.percentile(wlb_hi_happ_arr[:, 3], 100/3)
q67_wlb_hi_prf = np.percentile(wlb_hi_happ_arr[:, 3], 2*100/3)
wlb_hi_happ_low_perf = np.round(np.mean(wlb_hi_happ_arr[np.where((wlb_hi_happ_arr[:, 3] < q33_wlb_hi_prf))].T[3]), 1)
wlb_hi_happ_mid_perf = np.round(np.mean(wlb_hi_happ_arr[np.where((wlb_hi_happ_arr[:, 3] >= q33_wlb_hi_prf) & (wlb_hi_happ_arr[:, 3] < q67_wlb_hi_prf))].T[3]), 1)
wlb_hi_happ_hi_perf = np.round(np.mean(wlb_hi_happ_arr[np.where((wlb_hi_happ_arr[:, 3] >= q67_wlb_hi_prf))].T[3]), 1)
print(f'wlb_hi_happ_low_perf = {wlb_hi_happ_low_perf}')
print(f'wlb_hi_happ_mid_perf = {wlb_hi_happ_mid_perf}')
print(f'wlb_hi_happ_hi_perf = {wlb_hi_happ_hi_perf}')
# print(f'q33_wlb_low_prf = {q33_wlb_low_prf}')
# print(f'q67_wlb_low_prf = {q67_wlb_low_prf}')
# print(f'q33_wlb_med_prf = {q33_wlb_med_prf}')
# print(f'q67_wlb_med_prf = {q67_wlb_med_prf}')
# print(f'q33_wlb_hi_prf = {q33_wlb_hi_prf}')
# print(f'q67_wlb_hi_prf = {q67_wlb_hi_prf}')
cellText = [[f'( {wlb_hi_happ_hi_perf}, {whol_hi_happ_hi_perf} )',
                          f'( {wlb_hi_happ_mid_perf}, {whol_hi_happ_mid_perf} )',
                          f'( {wlb_hi_happ_low_perf}, {whol_hi_happ_low_perf} )'],
                         [f'( {wlb_med_happ_hi_perf}, {whol_med_happ_hi_perf} )',
                          f'( {wlb_med_happ_med_perf}, {whol_med_happ_mid_perf} )',
                          f'( {wlb_med_happ_low_perf}, {whol_med_happ_low_perf} )'],
                         [f'( {wlb_low_happ_hi_perf}, {whol_low_happ_hi_perf} )',
                          f'( {wlb_low_happ_mid_perf}, {whol_low_happ_mid_perf} )',
                          f'( {wlb_low_happ_low_perf}, {whol_low_happ_low_perf} )']]
print(cellText)
ax[1][1].table(cellText=cellText,
               rowLabels=['Happy', 'Medium', 'Unhappy'],
               colLabels=['Happy', 'Medium', 'Unhappy'],
               loc='center')
ax[1][1].set_title("Performance Benefit Comparison")
ax[1][1].axis('off')

plt.grid(True)
plt.tight_layout()
plt.savefig('./image/13GameTheoryIdealHoursPerWeek.png')
plt.show()
#