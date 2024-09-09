import numpy as np
import scipy.stats as sp

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

print(f'********* 2: Ex *********')
print(f'1. qnorm(.99, 1, sqrt(25)) = {sp.norm.ppf(.99, 1, np.sqrt(25))}')
print(sp.t.ppf(0.05, 22))
print(sp.chi2.ppf(0.05, 22))
print(sp.chi2.ppf(0.95, 22))

print((67*4.89*4.89 + 73*6.43*6.43)/(67+73))
print((26.99-35.76)/(np.sqrt(((1/68+1/74)*33.0020599999))))
print(26.99-35.76)
print(sp.t.ppf(0.05, 140))
print(sp.f.ppf(0.05, 27, 25))


print(f'****************************')

