import pandas as pd

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

yr = pd.Period('2024', freq='Y')
print(f'Properties of Period {yr}:\n{dir(yr)}')
print(f'Start time: {yr.start_time}\nEnd time: {yr.end_time}\n')

mo = pd.Period('2024-07', freq='M')
print(f'Properties of Period {mo}:\n{dir(mo)}')
print(f'Start time of next month: {(mo+1).start_time}\nEnd time of Sep 2024: {(mo+2).end_time}\n')

dy = pd.Period('2024-02-01', freq='D')
print(f'Properties of Period {dy}:\n{dir(dy)}')
print(f'Start time of next month: {(dy+29).start_time}\nEnd time of Mar 2024: {(dy+59).end_time}\n')

hr = pd.Period('2024-02-01 05:00:00', freq='h')
print(f'Start time of next hour: {hr+pd.offsets.Hour(1)}\nEnd time of today: {(hr+18).end_time}\n')

qr = pd.Period('2024Q1', freq='Q')
print(f'Period {qr}')
print(f'Start time of quarter: {qr.start_time}\nEnd time of this quarter: {qr.end_time}\n')

qr = pd.Period('2023Q2', freq='Q-JAN')    # Walmart Fiscal Feb - Jan Next yr
print(f'Period {qr}')
print(f'Start time 2 quarters later: {(qr+2).start_time}\nEnd time 3 quarters earlier: {(qr+3).end_time}\n')
