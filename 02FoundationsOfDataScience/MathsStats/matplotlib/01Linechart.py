import numpy as np
from matplotlib import pyplot as plt

# print(plt.style.available)
# plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
# plt.xkcd()  # xkcd comic style

exp_yrs_x = np.linspace(0, 25, 11)
# print(exp_yrs_x)
med_sal_usdk_y = np.array([83.6, 87.0, 92.6, 100.4, 104.5, 108.5, 112.6, 117.9, 120, 122.3, 125])
med_sal_usdk_IT_y = np.array([88.3, 92.7, 99.2, 100.4, 104.5, 113.8, 120.6, 123, 130, 132.3, 139])
med_sal_usdk_AI_y = np.array([92.9, 99.3, 106.2, 112.2, 119.9, 127.6, 136, 143.1, 150.4, 162.7, 179.5])

plt.title("Median Salary (K U$D) by Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Median Salary (K U$D)")

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

plt.plot(exp_yrs_x, med_sal_usdk_y, marker='d', linestyle='-.', color='k', label="Engineers")
plt.plot(exp_yrs_x, med_sal_usdk_AI_y, marker='o', linestyle=':', color='blue', label="AI")
plt.plot(exp_yrs_x, med_sal_usdk_IT_y, marker='*', linestyle='-', color='#ffbb77', label="IT")

         # marker='o', linestyle=':', color='#ffaf83', linewidth='3', label="AI")
         # marker='*', linestyle='-', color='g', label="IT")

plt.legend()
# plt.grid(True)
# plt.tight_layout()
plt.savefig('./plots/01Linechart.png')
plt.show()