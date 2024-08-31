import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

us_biz_days = CustomBusinessDay(calendar=USFederalHolidayCalendar())
print(f'\nus_biz_days = {us_biz_days} :: type(us_biz_days) = {type(us_biz_days)})')

date_range = pd.date_range(start='2024-07-01', end='2024-07-31', freq=us_biz_days)
print(f'\ndate_range = \n{date_range}\n'
      f'\n** Note that 2024-07-04 - Thursday is absent as 4-th July us US Holiday')
