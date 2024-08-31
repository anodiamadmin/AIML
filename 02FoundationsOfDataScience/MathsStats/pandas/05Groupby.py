import pandas as pd
import matplotlib.pyplot as plt

df_weather = pd.read_csv('./data/01weather.csv')
# print(df_weather)
df_weather_grouped = df_weather.groupby('weather_type')
# print(df_weather_grouped)

# for weather_type, weather_type_df in df_weather_grouped:
#     print(f'weather_type = {weather_type}')
#     print(weather_type_df)

# df_partly_cloudy = df_weather_grouped.get_group('Partly Cloudy')
# print(df_partly_cloudy)

# Split Apply Combine
# print(f'max precipitation chance as per weather types =\n{df_weather_grouped.precip_chnce.max()}')
# print(f'Average maximum temperature as per weather types =\n{df_weather_grouped.max_temp.mean()}')
# print(f'Minimum of minimum temperatures as per weather types =\n{df_weather_grouped.min_temp.min()}')

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
print(f'Descriptive statistics of all weather types\n{df_weather_grouped.describe()}')
print(f'Descriptive statistics of all weather types\n{df_weather_grouped.describe(include='all')}')