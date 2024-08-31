import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import count
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals, y_vals = [], []

index = count()

def animate(i):
    x_vals.append(next(index))
    y_vals.append(np.random.randint(1, 6))
    plt.cla()
    plt.plot(x_vals, y_vals)
    if len(x_vals) > 15:
        x_vals.pop(0)
        y_vals.pop(0)

anim = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show()