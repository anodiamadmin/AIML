import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_stocks = pd.read_csv('./data/NFLX.csv')
# print(f'Type of df_stocks[\'Date\'] as read from CSV                  = {type(df_stocks.Date[0])}')
df_stocks.Date = pd.to_datetime(df_stocks.Date)
# print(f'Type of df_stocks[\'Date\'] after converting to datetime      = {type(df_stocks.Date[0])}')
# print(f'df_stocks before setting Date as Index:\n{df_stocks.head(2)}')
df_stocks = df_stocks.set_index("Date")
# print(f'df_stocks AFTER setting Date as Index:\n{df_stocks.head(2)}')

# df_stocks_parse_date = pd.read_csv('./data/NFLX.csv', parse_dates=['Date'], index_col="Date")
# print(f'\nCSV reading + Date column conversion to type Timestamp and set as Index in one line:\n'
#       f'{df_stocks_parse_date.head(2)}')

# print(f'\nDate index of df_stocks:\n{df_stocks.index}')
# df_2019_07_03 = df_stocks['2019-07-03': '2019-07-03']
# print(f'\nData for 2019-07-03:\n{df_2019_07_03}')

# df_2021_03 = df_stocks['2021-03-01': '2021-03-31']
# df_2021_03 = df_2021_03.reset_index(level=0)
# print(f'\nTypeof df_2021_03.Date = {type(df_2021_03.Date[0])}')
# print(f'\nData for 2019-03:\n{df_2021_03}')

# df_date_range = df_stocks['2020-02-01': '2020-02-07']
# df_date_range_reverse = df_stocks['2020-02-07': '2020-02-01']
# print(f'\nData for date range:\n{df_date_range}')
# print(f'\nData for reverse date range is EMPTY:\n{df_date_range_reverse}')
