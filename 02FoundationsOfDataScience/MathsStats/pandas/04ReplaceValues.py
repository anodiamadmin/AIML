import numpy as np
import pandas as pd
df_weather = pd.read_csv('./data/04weather_Unclean.csv')
# print(df_weather)

# df_weather = df_weather.replace([0, -1, '-99', 100], 22)     # Replace all values from list [0.-1,-99,100] by 22
# df_weather = df_weather.replace(0, np.nan)    # OR np.NaN

# df_weather = df_weather.replace('[A-Za-z]', '', regex=True) # need to replace / and ' '
df_weather = df_weather.replace({'avg_temp': '[A-Za-z]', 'max_wind_speed': '[A-Za-z]'},
                                '', regex=True)
# Replace Mode of Wind Direction by int values from a list
df_weather = df_weather.replace({'wind_direction_mode': ['N', 'NNE', 'NE', 'NEE', 'E', '-9999']},
                                {'wind_direction_mode': [1, 2, 3, 4, 5, 0]})
print(df_weather)