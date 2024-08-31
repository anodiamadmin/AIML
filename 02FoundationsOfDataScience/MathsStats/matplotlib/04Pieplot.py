from matplotlib import pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

slices = np.array([59219, 55466, 47544, 36443, 35917])
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
colors = ['#f7a35c', '#e377c2', '#2ca02c', '#d62728', '#9467bd']
explode = [0, 0, 0, 0.1, 0.2]

plt.pie(slices, labels=labels, colors=colors, explode=explode, shadow=True,
        startangle=90, autopct='%1.2f%%', pctdistance=0.65,
        wedgeprops={'edgecolor': 'black'})

plt.title("Five Most Popular Programming Languages")
plt.tight_layout()
# plt.legend()
plt.savefig('./plots/04Pieplot.png')
plt.show()