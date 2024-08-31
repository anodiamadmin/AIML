import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
print(plt.style.available)

years = np.arange(2015, 2025)
corollaSales = np.arange(500, 2500, 200) + np.random.randint(0, 600, size=10)
print(f"corollaSales = {corollaSales}")
camrySales = np.arange(1400, 600, -80) + np.random.randint(0, 300, size=10)
print(f"camrySales = {camrySales}")
landCruiserSales = np.arange(300, 3300, 300) + np.random.randint(0, 800, size=10)
print(f"landCruiserSales = {landCruiserSales}")

labels = ['Camry', 'Corolla', 'Land Cruiser']

plt.stackplot(years, camrySales, corollaSales, landCruiserSales,
              labels=labels, colors=['#ff7f0e', '#1f77b4', '#2ca02c'])

plt.title("Toyota Sales in Last 10 Years")
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('./plots/05Stackplot.png')
plt.show()