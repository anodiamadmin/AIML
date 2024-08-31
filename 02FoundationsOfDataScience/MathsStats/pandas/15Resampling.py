import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
plt.style.use('ggplot')
fig, ax = plt.subplots(1, 1, figsize=(11, 5))
fig.suptitle('NFLX: Netflix, Inc: NasdaqGS', fontsize=16, color='#664422', font='Oxygen')

df_stocks = pd.read_csv('./data/NFLX.csv', parse_dates=['Date'], index_col="Date")
print(f'\nCSV reading + Date to type Timestamp and set as Index:\n{df_stocks.head(4)}')
ax.plot(df_stocks.Close, label='Daily Data', linewidth=.5, color='#224466', alpha=.7, linestyle=':')

# Select Monthly Mean of Close Column
df_close_monthly_mean = df_stocks.Close.resample('MS').mean()
# print(f'\nMonthly Mean of Close:\n{df_close_monthly_mean.head()}')
ax.plot(df_close_monthly_mean, label='Month Start Mean', linewidth=1, color='#ff5555', alpha=.8, linestyle='-')
# df_close_month_end_mean = df_stocks.Close.resample('ME').mean()
# ax.plot(df_close_month_end_mean, label='Month End Mean', linewidth=.7, color='#77aa33', alpha=.7, linestyle='--')

df_weekly = df_stocks.Close.resample('W').mean()

ax.plot(df_weekly, label='Weekly Mean', linewidth=.7, color='k', alpha=.7, linestyle='dotted')

df_quarterly_2021_22 = df_stocks['2021-01-01': '2022-12-31'].Close.resample('QS').mean()
ax.plot(df_quarterly_2021_22, label='Quarterly 2021-2022', color='blue', linestyle='--')

ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(pd.to_datetime('2019-06-10'), pd.to_datetime('2024-06-30'))
df_dates = df_stocks.Close.resample('QS').mean().reset_index()
# print(f'\ndf_stocks:\n{df_stocks}')
# print(f'\ndf_dates:\n{df_dates}')
ax.set_xticks(df_dates.Date)
ax.set_xticklabels(df_dates.Date.dt.to_period('Q').astype(str).str.replace('Q', '-Q'), rotation=90, fontsize=8)

plt.show()

# # Frequency
# # Alias    Description
# # B        business day frequency
# # C        custom business day frequency
# # D        calendar day frequency
# # W        weekly frequency
# # M        month end frequency
# # SM       semi-month end frequency (15th and end of month)
# # BM       business month end frequency
# # CBM      custom business month end frequency
# # MS       month start frequency
# # SMS      semi-month start frequency (1st and 15th)
# # BMS      business month start frequency
# # CBMS     custom business month start frequency
# # Q        quarter end frequency
# # BQ       business quarter end frequency
# # QS       quarter start frequency
# # BQS      business quarter start frequency
# # A, Y     year end frequency
# # BA, BY   business year end frequency
# # AS, YS   year start frequency
# # BAS, BYS business year start frequency
# # BH       business hour frequency
# # H        hourly frequency
# # T, min   minutely frequency
# # S        secondly frequency
# # L, ms    milliseconds
# # U, us    microseconds
# # N        nanoseconds
