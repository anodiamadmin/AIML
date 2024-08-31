import pandas as pd
import time
import datetime

print(f'time.time() = {time.time()}')               # Seconds since the Epoch Jan-1-1970 00:00:00
print(f'datetime.datetime.now() = {datetime.datetime.now()}')
print(f'time.ctime() = {time.ctime()}')
print(f'time.asctime() = {time.asctime()}')
print(f'time.strftime("%Y-%m-%d %H:%M:%S") = {time.strftime("%Y-%m-%d %H:%M:%S")}')
print(f'time.localtime() = {time.localtime()}')
print(f'time.gmtime() = {time.gmtime()}')
print(f'time.strptime("2024-07-04", "%Y-%m-%d") = {time.strptime("2024-07-04", "%Y-%m-%d")}')

t = 1820398163.2
print(f'\npd.to_datetime(t, unit=\'s\') = {pd.to_datetime(t, unit='s')}')  # default ns
t_arr = [1720398163.2, 1821398163.2, 1922398163.2, 1023398163.2]
dt = pd.to_datetime(t_arr, unit='s')
print(f'pd.to_datetime([t]) = {dt}')   # will return DatetimeIndex
print(f'dt.view(\'int64\') = {dt.view('int64')}')  # will return epoc time
