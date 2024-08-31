import pandas as pd

# Read a dataframe from .csv file
df_weather = pd.read_csv('./data/01weather.csv')

# Shape
# rows, columns = df_weather.shape  # shape returns a touple
# print(f'Data Contains: {rows} Rows and {columns} Columns')

# Head & Tail
# print(f'head(4), (default = 5 top rows) >>>>>\n{df_weather.head(4)}\n'
#       f'tail(4), (default = 5 bottom rows) >>>>>\n{df_weather.tail(4)}')

# Slicing
# print(f'Slicing 6th to 8th index rows >>>>>\n{df_weather[6:9]}')

# Show entire dataframe
# print(df_weather) # or # print(df_weather[:])

# Show column names
# print(df_weather.columns) # or # print(df_weather.columns.values)

# Show one or 2 columns
# print(df_weather['precip_chnce']) # or # print(df_weather[['precip_chnce']])
# print(df_weather[['precip_chnce', 'weather_type']])
# print(df_weather[['precip_chnce', 'weather_type']][6:13])

# Column names, data types, and number of non-null values
# print(df_weather.info())
# print(df_weather.describe())    # Stats of numeric columns only: count, mean, std, min, 25-50-75%, max
# print(df_weather.describe(include='all')) # Stats of non-numeric cols: unique, top, freq + stats of numeric colsof numeric cols
# print(df_weather.dtypes)        # Data types of columns
# print(type(df_weather))           # pandas.core.frame.DataFrame
# print(type(df_weather.precip_chnce)) # pandas.core.series.Series = print(type(df_weather['precip_chnce']))

# Statistics functions of columns for sanity check: count(), mean(), std(), min(), 25-50-75%, max()
# print(f'Max of max_temps = {df_weather['max_temp'].max()}')    # check for Outliers in Maximum Temp
# Row(s) with the highest temp (Hottest Day(s))
# print(df_weather[df_weather['max_temp'] == df_weather.max_temp.max()])
# print(df_weather[df_weather['max_temp'] == df_weather['max_temp'].max()]) # Syntax for col names with ' ' blank spaces
# list of col manes need to be passed to select specific cols
# print(df_weather[['date', 'max_temp', 'weather_type']][df_weather['max_temp'] == df_weather['max_temp'].max()])

# Set Reset Index # non int index => loc # default int index => iloc
# df_weather = df_weather.set_index('date')
# print(f'df_weather\n{df_weather}')
# print(f'df_weather of 01/06/2024\n{df_weather.loc["01/06/2024"]}')

# df_weather = df_weather.reset_index()
# print(f'df_weather\n{df_weather}')

df_weather = df_weather.set_index('weather_type')    # weather_type is NOT unique but it still works
print(f'df_weather of 01/06/2024\n{df_weather.loc["Sunny"]}')   # Returns all Sunny days
