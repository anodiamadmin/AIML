import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sp

plt.style.use('fivethirtyeight')
mu, sigma = 160.95, 5.59
heights = np.random.normal(mu, sigma, 100000)

bins = np.arange(mu-4*sigma, mu+4*sigma, .5)
plt.hist(heights, bins=bins, edgecolor='black', color='pink', linewidth=1)

# Mean and SD of Height
# Mean = Sum of all elements divided by number of elements
# SD = sqrt(sum of (x-mean)^2/n)
# Mode = Most frequent element
# Mode is not supported in np, se we use scipy.stats.mode
meanHeight = np.mean(heights)
sdHeight = np.std(heights)
approxModeHeight = np.bincount(heights.astype(int)).argmax()
spModeHeight = sp.mode(heights)[0]

plt.axvline(meanHeight+sdHeight, color='green', linestyle=':', linewidth=1, label='1SD Height')
plt.axvline(meanHeight-sdHeight, color='green', linestyle=':', linewidth=1)
plt.axvline(meanHeight+2*sdHeight, color='blue', linestyle=':', linewidth=1, label='2SD Height')
plt.axvline(meanHeight-2*sdHeight, color='blue', linestyle=':', linewidth=1)
plt.axvline(meanHeight+3*sdHeight, color='k', linestyle=':', linewidth=1, label='3SD Height')
plt.axvline(meanHeight-3*sdHeight, color='k', linestyle=':', linewidth=1)
plt.axvline(approxModeHeight, color='green', linestyle='-', linewidth=2, label='Approx Mode')
plt.axvline(spModeHeight, color='red', linestyle='-', linewidth=2, label='Wrong Mode')
plt.axvline(meanHeight, color='blue', linestyle='--', linewidth=1, label='Mean Height')

# bins = np.arange(mu+2*sigma, mu+4*sigma, .5)
# plt.hist(heights, bins=bins, edgecolor='black', color='pink', linewidth=1, log=True)

plt.title("Heights of 18YO Girls")
plt.xlabel("Height in cm")
plt.ylabel("Number of Girls")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/09BinomialMeanSDMode.png')
plt.show()