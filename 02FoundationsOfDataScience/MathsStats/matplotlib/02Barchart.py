import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

exp_yrs_x = np.linspace(0, 25, 11)
med_sal_usdk_y = [83.6, 87.0, 92.6, 100.4, 104.5, 108.5, 112.6, 117.9, 120, 122.3, 125]
med_sal_usdk_IT_y = [88.3, 92.7, 99.2, 100.4, 104.5, 113.8, 120.6, 123, 130, 132.3, 139]
med_sal_usdk_AI_y = [92.9, 99.3, 106.2, 112.2, 119.9, 127.6, 136, 143.1, 150.4, 162.7, 179.5]

x_indexes = np.arange(len(exp_yrs_x))
# print(x_indexes)
width = 0.25

plt.title("Median Salary (K U$D) by Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Median Salary (K U$D)")

plt.bar(x_indexes-width, med_sal_usdk_y, width=width, color='k', label="Engineers")
plt.bar(x_indexes, med_sal_usdk_AI_y, width=width, label="AI")
plt.bar(x_indexes+width, med_sal_usdk_IT_y, width=width, label="IT")

plt.legend()
print(f"x_indexes = ", x_indexes)
print(f"exp_yrs_x = ", exp_yrs_x)
plt.xticks(ticks=x_indexes,  labels=exp_yrs_x)
plt.savefig('./plots/02Barchart.png')
plt.show()