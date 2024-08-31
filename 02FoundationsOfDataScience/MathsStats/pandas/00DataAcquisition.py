import pandas as pd
import numpy as np
read_file = './data/00StockMarketData.csv'
write_file = './data/00StockMarketDataWrite.csv'
read_excel = './data/00StockMarketData.xlsx'
excel_worksheet = 'StockDataSheet'
write_excel = './data/00StockMarketDataWrite.xlsx'

# df_stock = pd.read_csv(read_file, skiprows=6)  # Skip 6 rows from top as meta/ header
# df_stock = pd.read_csv(read_file, header=6)    # Consider 7th row (6th index row) as header

# df_stock = pd.read_csv(read_file, header=None) # For .csv files without header info: Auto generate Col Names as 0 1 2 etc
# col_list = ['stock_ticker', 'eps', 'revenue', 'price', 'business_head']
# df_stock.columns = col_list
# df_stock = pd.read_csv(read_file, header=None, names=['stock_ticker', 'eps', 'revenue', 'price', 'business_head']) # Provide Col names as a list

# df_stock = pd.read_csv(read_file, header=6, nrows=3) # read first n rows excluding header

# na_val_list = ['not available', 'n.a.']
# df_stock = pd.read_csv(read_file, header=6, na_values=na_val_list)

na_val_dict = {
    'revenue': ['not available', 'n.a.', '-1'],
    'price': ['not available', 'n.a.'],
    'business_head': ['not available', 'n.a.']
}
df_stock = pd.read_csv(read_file, header=6, na_values=na_val_dict)
# print(f'df_stock\n{df_stock}')

# df_stock.to_csv(write_file) # By default writes index column
# df_stock.to_csv(write_file, index=False) # Avoids the index column
# df_stock.to_csv(write_file, index=False, header=False) # Avoids the header row as well
# print(f'df_stock.columns = {df_stock.columns}')
# df_stock.to_csv(write_file, columns=['stock_ticker', 'eps'], index=False)  # only writes the required 2 columns

# df_stock = pd.read_excel(read_excel, header=6, sheet_name=excel_worksheet)  # Worksheet to be specified
def convert_biz_head(cell_val):
    if cell_val == 'n.a.':
        return 'Sam Walton'
    return cell_val
def convert_rev(revenue):
    if int(revenue) < 0:
        return np.nan
    return revenue
# df_stock = pd.read_excel(read_excel, header=6, sheet_name=excel_worksheet, na_values=na_val_dict,
#                          converters={'business_head': convert_biz_head, 'revenue': convert_rev})
# # print(f'df_stock\n{df_stock}')
# df_stock.to_excel(write_excel, index=False, sheet_name='writing sheet', startrow=6, startcol=1)

# Create a dataframe from a dictionary
dict_weather = {
    'date': ['22/05/2024','23/05/2024','24/05/2024','25/05/2024','26/05/2024','27/05/2024','28/05/2024',
             '29/05/2024','30/05/2024','31/05/2024','01/06/2024','02/06/2024','03/06/2024','04/06/2024'],
    'max_temp': [18,19,19,19,19,20,20,19,20,20,18,17,18,20],
    'min_temp': [9,10,12,12,11,11,11,11,11,13,9,8,13,14],
    'precip_chnce': [2,6,5,62,24,22,14,6,7,71,100,100,2,0],
    'weather_type': ['Sunny','Sunny','Sunny','Cloudy','Partly Cloudy','Partly Cloudy','Partly Cloudy',
                     'Sunny','Rainy','Rainy','Rainy','Sunny','Sunny','Sunny']
}
df_weather = pd.DataFrame(dict_weather)
# print(f'Weather Dataframe from Dictionary\n{df_weather}')

with pd.ExcelWriter('./data/00Share_Weather.xlsx') as writer:
    df_stock.to_excel(writer, index=False, sheet_name='shares', startrow=6, startcol=1)
    df_weather.to_excel(writer, sheet_name='weather')