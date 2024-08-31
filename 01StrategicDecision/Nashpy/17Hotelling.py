import numpy as np
from matplotlib import pyplot as plt

hotel_start = 1
hotel_end = 100
μ = np.random.randint(hotel_start, hotel_end)

dist = np.random.poisson(μ, 1000)
print(dist)
print(len(dist))

h, bins = np.histogram(dist, bins=np.linspace(hotel_start, hotel_end, hotel_end-hotel_start), range=(hotel_start, hotel_end))
#print(h)
#print(bins)

# mean = np.mean(spkt)
# print(f"Mean value {mean} versus mu {μ}")
#
# # Poisson PMF for given mu
# x = [k for k in range(min, max+1)]
# y = [poisson.pmf(k, μ) for k in range(min, max+1)]
#
# # plot sampled vs computed PMF
plt.hist(dist, bins=bins, density=True)
# plt.plot(x, y, "ro")
plt.title("Poisson")
plt.show()