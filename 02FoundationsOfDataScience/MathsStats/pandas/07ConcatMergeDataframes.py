import pandas as pd

df_india_weather = pd.DataFrame({
    'location': ['New Delhi', 'Mumbai', 'Kolkata', 'Chennai'],
    'avg_temp': [26, 24, 24, 28],
    'yrly_rain_mm': [105, 176, 182, 166]
})

df_australia_weather = pd.DataFrame({
    'location': ['Sydney', 'Melbourne', 'Brisbane', 'Adelaide'],
    'avg_temp': [20, 17, 23, 18],
    'yrly_rain_mm': [162, 153, 182, 166]
})

# Concat as Stack (one on top of other) Default: Row wise => (axis = 0)
# df_weather = pd.concat([df_india_weather, df_australia_weather])
# df_weather = pd.concat([df_india_weather, df_australia_weather], keys=['India', 'Australia'])
# df_weather_aus = df_weather.loc['Australia']
# print(f'Weather of Australia\n{df_weather_aus}')
df_weather = pd.concat([df_india_weather, df_australia_weather], ignore_index=True)
# print(f'Weather of India and Australia\n{df_weather}')

# Concat Sideways (one beside another) Column wise => (axis = 1)
# df_min_max_temp = pd.DataFrame({
#     'location': ['New Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Sydney', 'Melbourne', 'Brisbane', 'Adelaide'],
#     'min_temp': [6, 19, 12, 20, 2, 5, 14, 4],
#     'max_temp': [46, 41, 43, 45, 39, 36, 42, 35]
# })
# df_weather_full = pd.concat([df_weather, df_min_max_temp], axis=1)
# print(f'Full Weather of India and Australia\n{df_weather_full}')

df_min_max_temp_uneven_index = pd.DataFrame({
    'min_temp': [2, 5, 14, 4, -6, 6, 19, 12],
    'max_temp': [39, 36, 42, 35, 13, 46, 41, 43]
}, index=[4, 5, 6, 7, 0, 1, 2, 8])
df_weather_indexed_full = pd.concat([df_weather, df_min_max_temp_uneven_index], axis=1)
df_weather_indexed_full['location'] = (df_weather_indexed_full['location']
                                       .replace(df_weather_indexed_full.iloc[8]['location'], 'Canada'))
# print(f'Full Weather of Australia, Canada & India:\n{df_weather_indexed_full}')

df_climate = pd.Series(
    { 0: 'Hot and Dry',
            1: 'Hot and Humid',
            2: 'Hot and Humid',
            3: 'Hot and Humid',
            8: 'Cold and Dry',
            4: 'Subtropical',
            5: 'Subtropical',
            6: 'Hot and humid',
            7: 'Subtropical'
}, name='climate')
df_weather_climate = pd.concat([df_weather_indexed_full, df_climate], axis=1)
# print(f'Climate and Weather of Australia, Canada & India:\n{df_weather_climate}')

# Merge
df_humidity = pd.DataFrame({
    'location': ['New Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'London', 'Singapore', 'Tokyo', 'New York', 'San Francisco'],
    'humidity': [62, 84, 89, 96, 72, 92, 71, 69, 88]
})
# df_climate_humidity = pd.merge(df_weather_climate, df_humidity, on='location') # defaul how='inner'
# df_climate_humidity = pd.merge(df_weather_climate, df_humidity, on='location', how='outer') # how='left'/ 'right'
df_climate_humidity = pd.merge(df_weather_climate, df_humidity, on='location', how='right', indicator=True)
print(f'Climate with Humidity:\n{df_climate_humidity}')

df_climate1 = pd.DataFrame({
    'location': ['New Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad'],
    'humidity': [62, 84, 89, 96, 71],
    'temperature': [26, 24, 24, 28, 28]
})
df_climate2 = pd.DataFrame({
    'location': ['New Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bangalore'],
    'humidity': [62, 84, 89, 96, 64],
    'temperature': [26, 24, 24, 28, 22]
})
df_climate_final = pd.merge(df_climate1, df_climate2, on='location', suffixes=('_hyd', '_blr'), how='outer')
print(f'df_climate_final\n{df_climate_final}')