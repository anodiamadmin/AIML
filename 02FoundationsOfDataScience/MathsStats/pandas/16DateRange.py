import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

nasdaqBusinessCalendar = CustomBusinessDay(calendar=USFederalHolidayCalendar())

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
fig.suptitle('NFLX: Netflix, Inc: NasdaqGS: Feb-2023', fontsize=16, color='#664422', font='Oxygen')

df_stocks = pd.read_csv('./data/NFLX2023FebNoDates.csv')
# print(f'\nReading CSV data:\n{df_stocks}')
# date_rng = pd.date_range(start='1-Feb-2023', end='28/2/2023',    # Takes all date formats
#                          freq=nasdaqBusinessCalendar)     # Custom Frequency
                         # freq='B')     # B = Biz Days; D = Days; For more options lookup table below
date_rng = pd.date_range(start='1-Feb-2023', periods=19,    # Date Range with 19 periods
                         freq=nasdaqBusinessCalendar)       # Can create date range in both ways
print(f'\ndate_rng = {date_rng}')
df_stocks = df_stocks.set_index(date_rng)
# print(f'\ndf_stocks AFTER setting Date Range as Index:\n{df_stocks.head(2)}')

df_stocks.Close.plot(ax=ax, label='Business Day Data Feb 2023 (B)', linewidth=.5, color='#224466', alpha=.7, linestyle=':')

df_1_10_feb_2023 = df_stocks['2023-02-01': '2023-02-10']
# print(f'\nMean Close value for 1st to 10th Feb 2023 = {df_1_10_feb_2023.Close.mean()}')
df_1_10_feb_2023.Close.plot(ax=ax, label='1st to 10th Feb 2023', linewidth=1, color='Red', alpha=.7, linestyle='-')

df_stocks_all_days = df_stocks.asfreq('D', method='pad')
# print(f'\ndf_stocks_all_days :\n{df_stocks_all_days}')
df_stocks_all_days.Close.plot(ax=ax, label='Daily Data Feb 2023 (D)', linewidth=.5, color='#224466', alpha=.7, linestyle='--')

df_stocks_weekly = df_stocks.asfreq('W', method='pad')
df_stocks_weekly.Close.plot(ax=ax, label='Weekly Data Feb 2023 (W)', linewidth=.5, color='blue', alpha=.7, linestyle='--')

df_stocks_hourly = df_stocks.asfreq('h', method='pad')
df_stocks_hourly.Close.plot(ax=ax, label='Hourly Data Feb 2023 (h)', linewidth=.5, color='green', alpha=.8, linestyle='-')
print(f'Hourly price:\n{df_stocks_hourly}')

ax.legend(loc='lower left', fontsize=8)
df_dates = df_stocks.Close.resample('W').mean()
ax.set_xticks(df_dates.index)
ax.set_xlim(pd.to_datetime('2023-01-29'), pd.to_datetime('2023-03-03'))
plt.show()
