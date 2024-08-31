import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
mu, sigma = 160.95, 5.59
heights = np.random.normal(mu, sigma, 100000)

bins = np.arange(mu-4*sigma, mu+4*sigma, .5)
plt.hist(heights, bins=bins, edgecolor='black', color='pink', linewidth=1)

meanHeight = np.mean(heights)
sdHeight = np.std(heights)
q_25 = np.quantile(heights, 0.25)
q_50 = np.quantile(heights, 0.5) # Median
q_75 = np.quantile(heights, 0.75)

plt.axvline(meanHeight, color='red', linestyle='--', linewidth=1, label='Mean')
plt.axvline(meanHeight+sdHeight, color='green', linestyle=':', linewidth=1, label='SD')
plt.axvline(meanHeight-sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(meanHeight+2*sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(meanHeight-2*sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(meanHeight+3*sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(meanHeight-3*sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(q_25, color='blue', linestyle='--', linewidth=1, label='1 Qunt')
plt.axvline(q_50, color='k', linestyle='--', linewidth=1, label='2 Qunt/ Mode')
plt.axvline(q_75, color='blue', linestyle='--', linewidth=1, label='3 Qunt')

plt.title("Heights of 18YO Girls")
plt.xlabel("Height in cm")
plt.ylabel("Number of Girls")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/11MeanSDvsModeQntl.png')
plt.show()