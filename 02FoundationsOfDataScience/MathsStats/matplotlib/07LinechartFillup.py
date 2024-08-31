import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

plt.title("India vs Australia")
plt.xlabel("Overs")
plt.ylabel("Runs")

runsInd = np.cumsum(np.append(np.append(np.random.randint(0, 15, size=15),
                           np.random.randint(1, 10, size=25)),
                 np.random.randint(3, 22, size=10)))
runsAus = np.cumsum(np.append(np.append(np.random.randint(0, 15, size=15),
                           np.random.randint(1, 10, size=25)),
                 np.random.randint(3, 20, size=10)))
overs = np.arange(1, 51)
plt.plot(overs, runsInd, color='#469afc', linestyle='-', label="India")
plt.plot(overs, runsAus, color='#46fc9a', linestyle='-', label="Australia")
plt.fill_between(overs, runsInd, runsAus,
                 where=(runsInd > runsAus), interpolate=True,
                 color='red', alpha=0.2, label='India Winning')
plt.fill_between(overs, runsInd, runsAus,
                 where=(runsInd < runsAus), interpolate=True,
                 color='k', alpha=0.2, label='Australia Winning')

plt.legend()
plt.grid(True)
plt.tight_layout()
x_ticks = np.arange(10, 51, 10)
# print(f"x_ticks = ", x_ticks)
plt.xticks(ticks=x_ticks,  rotation=-45)
plt.savefig('./plots/07LinechartFillup.png')
plt.show()