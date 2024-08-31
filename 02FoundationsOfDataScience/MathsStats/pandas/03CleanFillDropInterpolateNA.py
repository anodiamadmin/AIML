import numpy as np
import pandas as pd

# df_weather = pd.read_csv('./data/01weather.csv')
# print(f'Dates are strings: datatype of date column = {type(df_weather.date[0])}')

# df_weather = pd.read_csv('./data/03weather_unclean.csv')
# print(f'Datatype of date column is String = {type(df_weather.date[0])}')
df_weather = pd.read_csv('./data/03weather_unclean.csv', parse_dates=['date'])
# print(f'Datatype of date column is Timestamp (converted from String) = {type(df_weather.date[0])}')
# print(df_weather)
df_weather = df_weather.set_index('date')
# print(df_weather)
# df_weather = df_weather.fillna(0)
# print(df_weather)

# df_weather = df_weather.dropna()        # lot of columns lost, instead try interpolating those values
# df_weather = df_weather.dropna(how='all')       # when all fields sans the index col are NaN
# df_weather = df_weather.dropna(thresh=1)  # when more than the threshold of 1 non index field are NaN

# df_weather = df_weather.fillna({
#     # 'max_temp': 0,
#     # 'min_temp': df_weather.min_temp.min(),
#     'precip_chnce': df_weather.min_temp.mean(),
#     # 'weather_type': 'Unpredictable'
# })
# # df_weather = df_weather.fillna({'max_temp': 21, 'min_temp': 6})
# # df_weather['max_temp'] = df_weather['max_temp'].fillna(22)
# # df_weather['min_temp'] = df_weather['min_temp'].ffill(axis=0, limit=2)     # df.fillna(ffill) deprecated
# # df_weather['weather_type'] = df_weather['weather_type'].bfill(axis='rows') # axis='columns' removed
# # https://pandas.pydata.org/docs/
# df_weather = df_weather.interpolate(method='linear') # Direct use of df.interpolate is deprecated
# # df_weather = df_weather.infer_objects(copy=False).interpolate(method='time') # uses datetime index as the basis of interpolation

df_daterange = pd.date_range('2024-05-22', '2024-06-04')
date_index = pd.DatetimeIndex(df_daterange)
df_weather = df_weather.reindex(date_index) # use fillna() functions to fill in values for 2024-05-24 & 2024-05-25

print(df_weather)