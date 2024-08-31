import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn-v0_8')

# # Create Endurance CSV Data
##############################
# dates = pd.date_range('2023-01-01', periods=30, freq='D').strftime("%Y-%m-%d")
# distance = np.random.poisson(10, 30)
# heartRate = np.random.normal(140, 10, 30).round().astype(int)
# avgPace = np.random.normal(360, 10, 30).round().astype(int)
# endurance_data = np.transpose(np.array([dates, distance, heartRate, avgPace]))
# df_endurance = pd.DataFrame(endurance_data, columns=['Date', 'Distance', 'HeartRate', 'AvgPace'])
# df_endurance.to_csv('./data/endurance.csv', index=False)

# # Plot Social Media CSV Data
# ############################
df = pd.read_csv('./data/endurance.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
dates, distance, heartRate, avgPace = df['Date'].to_numpy(), df['Distance'].to_numpy(), df['HeartRate'].to_numpy(), df['AvgPace'].to_numpy()
# print(f"Date: {dates}, distance = {distance}, heartRate = {heartRate}, avgPace = {avgPace}")
plt.plot_date(dates, distance, linestyle='solid', linewidth=1, color='pink')
plt.title("Endurance Data")
plt.xlabel("Date")
plt.ylabel("Daily Kms")
plt.legend(['Distance'])
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%Y, %b %d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('./plots/21DateTimeCSV.png')
plt.show()
