import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba

# Load data
file_path = "GlobalWarmingAbove20thCentury.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Prepare data
x_data = df['Year'].values
y_data = df['Anomaly'].values
x0 = x_data.min()
x_norm = x_data - x0

# Define exponential model: y = a * exp(b * x)
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Fit exponential curve
params, _ = curve_fit(exp_func, x_norm, y_data, p0=(0.01, 0.01), maxfev=10000)
a_fit, b_fit = params

# Generate regression x/y
x_fit_full = np.linspace(x_data.min(), x_data.max(), 500)
y_fit_full = exp_func(x_fit_full - x0, a_fit, b_fit)

# Color interpolation from skyblue (1940) to red (1970)
def interpolate_color(year):
    start_year, end_year = 1940, 1970
    if year <= start_year:
        return to_rgba('skyblue')
    elif year >= end_year:
        return to_rgba('red')
    else:
        t = (year - start_year) / (end_year - start_year)
        sky = np.array(to_rgba('skyblue'))
        red = np.array(to_rgba('red'))
        return tuple((1 - t) * sky + t * red)

# Setup plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Global Temperature Anomalies Over the Years", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.set_xlim(x_data.min() - 1, x_data.max() + 1)
ax.set_ylim(min(y_data.min(), y_fit_full.min()) - 0.2, max(y_data.max(), y_fit_full.max()) + 0.2)
ax.grid(True)

# Background color zones
ax.axhspan(0, y_data.max() + 0.5, facecolor='#ffdddd', zorder=0)
ax.axhspan(y_data.min() - 0.5, 0, facecolor='#ddeeff', zorder=0)

# Bar container
bars = ax.bar(x_data, np.zeros_like(y_data), width=1.0, color='grey', zorder=2)

# Initialize regression line
regression_line, = ax.plot([], [], linewidth=6, zorder=4)

# Init
def init():
    for bar in bars:
        bar.set_height(0)
    regression_line.set_data([], [])
    return list(bars) + [regression_line]

# Update frame
def update(frame):
    # Update bars
    for i in range(frame + 1):
        height = y_data[i]
        bars[i].set_height(height)
        bars[i].set_color('darkred' if height >= 0 else 'darkblue')

    # Regression line progression
    current_year = x_data[min(frame, len(x_data) - 1)]
    x_fit = x_fit_full[x_fit_full <= current_year]
    y_fit = exp_func(x_fit - x0, a_fit, b_fit)

    regression_line.set_data(x_fit, y_fit)
    regression_line.set_color(interpolate_color(current_year))

    return list(bars) + [regression_line]

# Add 5-second pause (125 frames at 25 FPS)
pause_frames = 125
total_frames = len(x_data) + pause_frames

def extended_update(frame):
    if frame < len(x_data):
        return update(frame)
    else:
        return update(len(x_data) - 1)

# Animate
ani = animation.FuncAnimation(
    fig,
    extended_update,
    frames=total_frames,
    init_func=init,
    blit=False,
    interval=40,
    repeat=True  # set to False, to stop after one loop
)

# Save animation
plt.draw()
ani.save("temperature_anomalies_bars_with_exp_line.gif", writer='pillow', fps=25)
plt.close(fig)
