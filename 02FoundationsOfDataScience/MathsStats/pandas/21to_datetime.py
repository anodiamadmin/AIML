import pandas as pd

dates = ['4-Dec-2024', '20-Nov-2024', '06-Jul-2016']
date_indexes = pd.to_datetime(dates)
print(f'\ndate_indexes = \n{date_indexes}')

# Errors = Ignore Or Coerce
dates_err = ['4-Dec-2024', '20-Nov-2024', 'abc']
# date_indexes_err_ignore = pd.to_datetime(dates_err, errors='ignore')   # errors='ignore' is deprecated
# print(f'\ndate_indexes_err_ignore = \n{date_indexes_err_ignore}')      # ignores all dates
# Default errors='raise'
date_indexes_err_coerce = pd.to_datetime(dates_err, errors='coerce')  # ignores only wrong string 'abc'
print(f'\ndate_indexes_err_coerce = \n{date_indexes_err_coerce}')

date_times = ['4-Dec-2024 13:30:00', '20-Nov-2024 21:20:00', '06-Jul-2016 07:02:00']
date_time_indexes = pd.to_datetime(date_times)
print(f'\ndate_time_indexes = \n{date_time_indexes}')

us_dates = ['4/12/2024', '12/4/2004']
us_date_indexes = pd.to_datetime(us_dates)
print(f'\nus_date_indexes = \n{us_date_indexes}')

non_us_dates = ['4/12/2024', '12/4/2004']
non_us_date_indexes = pd.to_datetime(non_us_dates, dayfirst=True)
print(f'\nnon_us_date_indexes = \n{non_us_date_indexes}')

formatted_dates = ['4-Dec-2024', '12-Apr-2004']
formatted_date_indexes = pd.to_datetime(formatted_dates, format='%d-%b-%Y')
print(f'\nformatted_date_indexes = \n{formatted_date_indexes}')

custom_formatted_date_times = ['4$Dec$2024-12:30', '12$Apr$2004-02:00']
custom_formatted_date_time_indexes = pd.to_datetime(custom_formatted_date_times, format='%d$%b$%Y-%H:%M')
print(f'\ncustom_formatted_date_time_indexes = \n{custom_formatted_date_time_indexes}')
