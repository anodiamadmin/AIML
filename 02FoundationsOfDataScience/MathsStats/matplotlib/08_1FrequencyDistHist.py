import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

rollNums = np.arange(1, 51)
marks = np.random.randint(0, 11, size=50)

# print(f"Roll Numbers: {rollNums}")
# print(f"Marks: {marks}")

fig, ax = plt.subplots(1, 3, sharex=False, sharey=True)
fig.set_figheight(4)
fig.set_figwidth(12)

ax[0].bar(rollNums, marks, color='green', label="Marks")
ax[0].set_title("Marks of Students")
ax[0].set_xlabel("Roll Number")
ax[0].set_ylabel("Marks")
ax[0].legend(loc='upper left')

ax[1].plot(rollNums, marks, marker='o', linestyle='-', color='green', label="Marks", linewidth=1)
ax[1].set_title("Marks of Students")
ax[1].set_xlabel("Roll Number")
ax[1].set_ylabel("Marks")
ax[1].legend(loc='upper left')

# # bins = np.arange(0, 12, 2)
# # bins = np.arange(0, 12, 3)
# # bins = np.arange(0, 12, 5)
bins = np.arange(0, 11, 1)
ax[2].hist(marks, bins=bins, edgecolor='purple', color='blue', linewidth=1)
ax[2].set_title("Frequency Distribution of Marks")
ax[2].set_xlabel("Marks")
ax[2].set_ylabel("Number of Students")
ax[2].set_xlim([0, 11])

plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/08_1FrequencyDistHist.png')
plt.show()