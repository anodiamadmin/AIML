import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import count
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

def animate(i):
    live_df = pd.read_csv('./data/autoUpdate.csv').tail(50)
    x_value = live_df['x_value']
    y1_value = live_df['total1']
    y2_value = live_df['total2']
    plt.cla()
    plt.plot(x_value, y1_value, label='Channel1')
    plt.plot(x_value, y2_value, label='Channel2')
    plt.legend(loc='upper left')
    plt.tight_layout()

anim = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()