import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
fig.suptitle('Test Data Created', fontsize=16, color='#664422', font='Oxygen')

length = 100
date_rng = pd.date_range(start='1-Feb-2023', periods=length, freq='B')
# print(f'\ndate_rng = {date_rng}')

random_data = np.random.randint(1, 100, length)
time_series = pd.Series(random_data.cumsum(), index=date_rng, name='My_Data')
print(f'\ntime_series.head(5)::\n{time_series.head(5)}\ntype(time_series)={type(time_series)}')
time_series.plot(ax=ax, label='My Test Data', linewidth=1, color='Red', alpha=.7, linestyle='-')

ax.legend(loc='lower right', fontsize=8)
# ax.set_xticks(time_series.asfreq('MS', method='pad'))
ax.set_xlim(pd.to_datetime('2023-01-20'), pd.to_datetime('2023-06-28'))
plt.show()
