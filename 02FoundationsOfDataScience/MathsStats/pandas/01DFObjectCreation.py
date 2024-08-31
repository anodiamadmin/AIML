# $>> pip install pandas --upgrade
import pandas as pd

read_write_file = './data/01weather.csv'

# # # Create a dataframe from a dictionary
# dict_weather = {
#     'date': ['22/05/2024','23/05/2024','24/05/2024','25/05/2024','26/05/2024','27/05/2024','28/05/2024',
#              '29/05/2024','30/05/2024','31/05/2024','01/06/2024','02/06/2024','03/06/2024','04/06/2024'],
#     'max_temp': [18,19,19,19,19,20,20,19,20,20,18,17,18,20],
#     'min_temp': [9,10,12,12,11,11,11,11,11,13,9,8,13,14],
#     'precip_chnce': [2,6,5,62,24,22,14,6,7,71,100,100,2,0],
#     'weather_type': ['Sunny','Sunny','Sunny','Cloudy','Partly Cloudy','Partly Cloudy','Partly Cloudy',
#                      'Sunny','Rainy','Rainy','Rainy','Sunny','Sunny','Sunny']
# }
# df_weather = pd.DataFrame(dict_weather)
# print(f'Weather Dataframe from Dictionary\n{df_weather}')

# # Create a dataframe from a (Tuple of Tuples), [List of Tuples] or {Set of Tuples}
# tuple_weather = (        #  [        #  {
#     ('22/05/2024',18,9,2,'Sunny'),
#     ('23/05/2024',19,10,6,'Sunny'),
#     ('24/05/2024',19,12,5,'Sunny'),
#     ('25/05/2024',19,12,62,'Cloudy'),
#     ('26/05/2024',19,11,24,'Partly Cloudy'),
#     ('27/05/2024',20,11,22,'Partly Cloudy'),
#     ('28/05/2024',20,11,14,'Partly Cloudy'),
#     ('29/05/2024',19,11,6,'Sunny'),
#     ('30/05/2024',20,11,7,'Rainy'),
#     ('31/05/2024',20,13,71,'Rainy'),
#     ('01/06/2024',18,9,100,'Rainy'),
#     ('02/06/2024',17,8,100,'Sunny'),
#     ('03/06/2024',18,13,2,'Sunny'),
#     ('04/06/2024',20,14,0,'Sunny')
# )        #  ]        #  }
# # need to give column names to the dataframe separately as a list
# column_names = ['date', 'max_temp', 'min_temp', 'precip_chnce', 'weather_type']
# df_weather = pd.DataFrame(tuple_weather, columns=column_names)
# print(f'Weather Dataframe from Tuple, (List or Set) of Tuples\n{df_weather}')
# # Save dataframe to a .csv file
# df_weather.to_csv(read_write_file, index=False)

# Create a dataframe from a List of Dictionaries
weather_record = [
    {'date': '22/05/2024', 'max_temp': 18,'min_temp': 9, 'precip_chnce': 2, 'weather_type': 'Sunny'},
    {'date': '23/05/2024', 'max_temp': 19, 'min_temp': 10, 'precip_chnce': 6, 'weather_type': 'Sunny'},
    {'date': '24/05/2024', 'max_temp': 19, 'min_temp': 12, 'precip_chnce': 5, 'weather_type': 'Sunny'},
    {'date': '25/05/2024', 'max_temp': 19, 'min_temp': 12, 'precip_chnce': 62, 'weather_type': 'Cloudy'},
    {'date': '26/05/2024', 'max_temp': 19, 'min_temp': 11, 'precip_chnce': 24, 'weather_type': 'Partly Cloudy'},
    {'date': '27/05/2024', 'max_temp': 20, 'min_temp': 11, 'precip_chnce': 22, 'weather_type': 'Partly Cloudy'},
    {'date': '28/05/2024', 'max_temp': 20, 'min_temp': 11, 'precip_chnce': 14, 'weather_type': 'Partly Cloudy'},
    {'date': '29/05/2024', 'max_temp': 19, 'min_temp': 11, 'precip_chnce': 6, 'weather_type': 'Sunny'},
    {'date': '30/05/2024', 'max_temp': 20, 'min_temp': 11, 'precip_chnce': 7, 'weather_type': 'Rainy'},
    {'date': '31/05/2024', 'max_temp': 20, 'min_temp': 13, 'precip_chnce': 71, 'weather_type': 'Rainy'},
    {'date': '01/06/2024', 'max_temp': 18, 'min_temp': 9, 'precip_chnce': 100, 'weather_type': 'Rainy'},
    {'date': '02/06/2024', 'max_temp': 17, 'min_temp': 8, 'precip_chnce': 100, 'weather_type': 'Sunny'},
    {'date': '03/06/2024', 'max_temp': 18, 'min_temp': 13, 'precip_chnce': 2, 'weather_type': 'Sunny'},
    {'date': '04/06/2024', 'max_temp': 20, 'min_temp': 14, 'precip_chnce': 0, 'weather_type': 'Sunny'}
]
df_weather = pd.DataFrame(weather_record)
print(f'Weather Dataframe from Tuple, (List or Set) of Tuples\n{df_weather}')

# Read a dataframe from .csv file
# df_weather_read = pd.read_csv(read_write_file)
# print(f'Weather Dataframe\n{df_weather_read}')

# Read a dataframe from Excel file
# $>> pip install openpyxl --upgrade
# df_weather_excel = pd.read_excel('./data/01weatherExcel.xlsx', 'Sheet01weather')
# print(f'Weather Dataframe from Excel\n{df_weather_excel}')

# # Save dataframe to a .csv file
df_weather.to_csv(read_write_file, index=False)
