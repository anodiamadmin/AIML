import pandas as pd
pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
df_weather = pd.read_csv('./data/09melt.csv')
# print(df_weather)

# Melt is opposite of pivot
# df_weather_melt = pd.melt(df_weather, id_vars=['date', 'day'])
# print(df_weather_melt)
# filter one city out
# df_weather_melt = df_weather_melt[df_weather_melt['variable'] == 'Sydney']
# print(df_weather_melt)

# Melt with user defined var_name & value_name
df_weather = pd.melt(df_weather, id_vars=['date'], var_name='city', value_name='max_temp')
# print(df_weather)
# filter one city out
df_weather = df_weather[df_weather['city'] == 'Sydney']
print(df_weather)
