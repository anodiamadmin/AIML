import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn-v0_8')
# print(plt.style.available)

dates = [datetime(2024, 4, 24), datetime(2024, 4, 25),
         datetime(2024, 4, 26), datetime(2024, 4, 27),
         datetime(2024, 4, 28), datetime(2024, 4, 29),
         datetime(2024, 4, 30), datetime(2024, 5, 1)]
y = np.random.randint(0, 10, len(dates))

plt.plot_date(dates, y, linestyle='solid', linewidth=1, color='pink')
plt.title("Date Plot")
plt.xlabel("Date")
plt.ylabel("Y")
plt.legend(['y-vals'])
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%b, %d %Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('./plots/20DateTime.png')
plt.show()
