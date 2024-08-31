import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
print(plt.style.available)

marks = np.array([3, 1, 5, 0, 5, 5, 2, 3, 2, 5, 5, 4, 2, 3, 3, 4, 4, 4, 1, 4,
                  5, 6, 7, 10, 9, 6, 6, 6, 6, 9, 7, 7, 7, 8, 8, 8])
bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
print(bins)

plt.hist(marks, bins=bins, edgecolor='black', color='#ff55dd', linewidth=1)
plt.title("Histogram: Marks of 20 students")
plt.xlabel("Mark")
plt.ylabel("Frequency of Mark")

#
# df_runs = pd.read_csv('./data/ipl_2023_runs.csv')
# print(df_runs.head())
# Sixes = df_runs['Sixes']
# bins = np.arange(0, 50, 1)
# print(f'bins = {bins}')
# plt.hist(Sixes, bins=bins, edgecolor='black', color='#ff55dd', linewidth=1)
# plt.title("6s by Players in IPL 2023")
# plt.xlabel("6s")
# plt.ylabel("Frequency of Players")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/08Histogram.png')
plt.show()


