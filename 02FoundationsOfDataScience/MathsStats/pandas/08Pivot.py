import pandas as pd
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
# df_weather = pd.read_csv('./data/08weatherPivot.csv')
#
# df_weather = df_weather.pivot(index='date', columns='city', values=['max_temp', 'min_temp'])
# print(df_weather)
# print(df_weather['max_temp'])
#
# df_weather2 = pd.read_csv('./data/08weatherPivot2.csv')
# # print(df_weather2)
# # get average temp & Chance of precipitation
# df_weather2 = df_weather2.pivot_table(index='date', columns='city',
#                                       values=['temperature', 'precip_chnce'], aggfunc=['mean', 'sum'],
#                                       margins=True)
# print(df_weather2)

df_weather3 = pd.read_csv('./data/08weatherPivot3.csv')
# print(df_weather3.date.dtype)
df_weather3['date'] = pd.to_datetime(df_weather3['date'], dayfirst=True)
# print(df_weather3.date.dtype)
df_weather3 = df_weather3.pivot_table(index=pd.Grouper(freq='ME', key='date'), columns='city', values=['temperature'])
print(df_weather3)
