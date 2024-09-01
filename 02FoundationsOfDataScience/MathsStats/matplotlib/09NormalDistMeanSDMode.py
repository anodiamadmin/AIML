import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as sp

plt.style.use('fivethirtyeight')
mu, sigma = 160.95, 5.59
heights = np.random.normal(mu, sigma, 100000)

p_47 = sp.norm.cdf(0.47)
print(f'scipy.stats.norm.cdf(0.47) = {p_47}')   # R equivalent of:: pnorm(0.47) = 0.6808224912174442
q_68 = sp.norm.ppf(0.68)
print(f'scipy.stats.norm.ppf(0.68) = {q_68}')   # R equivalent of:: qnorm(0.68) = 0.4676987991145084

p_3_3 = sp.norm.cdf(3) - sp.norm.cdf(-3)   # R equivalent of:: pnorm(3) - pnorm(-3) = 0.9973002039367398
print(f'scipy.stats.norm.cdf(3) - sp.norm.cdf(-3) = {p_3_3}')

mu1, sigma1 = 10, np.sqrt(9)
p_non_standard_12_10_9 = sp.norm.cdf(12, loc=mu1, scale=sigma1)    # R equivalent of:: pnorm(12, mean=10, sd=3)
print(f'scipy.stats.norm.cdf(12, loc=mu, scale=sigma) = {p_non_standard_12_10_9}')

print(f'********* 1.7: Ex3 *********')
print(f'1. pnorm(-1, -2, sqrt(9)) = {sp.norm.cdf(-1, -2, np.sqrt(9))}')
print(f'2. 1 - pnorm(-3.5, -2, sqrt(9)) = {1 - sp.norm.cdf(-3.5, -2, np.sqrt(9))}')
print(f'3. pnorm(6, -2, sqrt(9) - pnorm(-7, -2, sqrt(9)) = '
      f'{sp.norm.cdf(6, -2, np.sqrt(9)) - sp.norm.cdf(-7, -2, np.sqrt(9))}')
print(f'****************************')

full_range = 95/100
bottom_range = (1-full_range)/2
top_range = 1-bottom_range
print(f'bottom_range = {bottom_range}, top_range = {top_range}, range = {top_range - bottom_range}')
mu2, sigma2 = 174, 6.4
q_bottom = sp.norm.ppf(bottom_range, loc=mu2, scale=sigma2)
q_top = sp.norm.ppf(top_range, loc=mu2, scale=sigma2)
print(f'{full_range * 100}% of heights is within {q_bottom} cm to {q_top} cm')

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