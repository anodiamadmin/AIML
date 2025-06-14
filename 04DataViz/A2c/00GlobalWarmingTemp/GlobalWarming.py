import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba
import matplotlib.image as mpimg

# Load data
file_path = "GlobalWarmingAbove20thCentury.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Extract and normalize
x_data = df['Year'].values
y_data = df['Anomaly'].values
emissions_raw = df['Emission'].values  # in million tons
emissions = emissions_raw / 1000       # convert to billion tons

x0 = x_data.min()
x_norm = x_data - x0

# Exponential model
def shifted_exp_func(x, a, b, c):
    return c + a * np.exp(b * x)

# Fit curve
params, _ = curve_fit(shifted_exp_func, x_norm, y_data, p0=(0.01, 0.01, -0.2), maxfev=10000)
a_fit, b_fit, c_fit = params

# Generate smooth curve
x_fit_full = np.linspace(x_data.min(), x_data.max(), 500)
y_fit_full = shifted_exp_func(x_fit_full - x0, a_fit, b_fit, c_fit)

# Color interpolation
def interpolate_color(year):
    start, end = 1940, 1970
    if year <= start:
        return to_rgba('skyblue')
    elif year >= end:
        return to_rgba('red')
    else:
        t = (year - start) / (end - start)
        return tuple((1 - t) * np.array(to_rgba('skyblue')) + t * np.array(to_rgba('red')))

# Load background image
bg_img = mpimg.imread("GlobalWarming.png")

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.figimage(bg_img, xo=0, yo=0, alpha=0.2, zorder=0)

# Twin axes
ax2 = ax1.twinx()

# Labels with proper CO₂ subscript
ax1.set_title("Global CO$_2$ Emissions vs Temperature Anomalies Over the Years", fontsize=14)
ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature Anomaly (°C)")
ax2.set_ylabel("CO$_2$ Emissions (Billion Tons)")

# Axes limits
ax1.set_xlim(x_data.min() - 1, x_data.max() + 1)
ax1.set_ylim(min(y_data.min(), y_fit_full.min()) - 0.2, max(y_data.max(), y_fit_full.max()) + 0.2)
ax2.set_ylim(-19.25, 40)

# Background zones
ax1.axhspan(0, y_data.max() + 0.5, facecolor='#ffdddd', zorder=1)
ax1.axhspan(y_data.min() - 0.5, 0, facecolor='#ddeeff', zorder=1)

# Temperature bars
bars = ax1.bar(x_data, np.zeros_like(y_data), width=1.0, color='grey', zorder=2)

# Regression and emission lines
regression_line, = ax1.plot([], [], linewidth=6, zorder=4)
emission_line, = ax2.plot([], [], color='darkgrey', linewidth=6, zorder=3)

# Emission tick formatting
def format_emission_ticks(y, _):
    return f"{y:.0f}" if y > 0 else ''

ax2.set_yticks(np.linspace(-15, 40, 12))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_emission_ticks))

# Initialization function
def init():
    for bar in bars:
        bar.set_height(0)
    regression_line.set_data([], [])
    emission_line.set_data([], [])
    return list(bars) + [emission_line, regression_line]

# Frame update function
def update(frame):
    for i in range(frame + 1):
        bars[i].set_height(y_data[i])
        bars[i].set_color('darkred' if y_data[i] >= 0 else 'darkblue')

    current_year = x_data[min(frame, len(x_data) - 1)]
    x_fit = x_fit_full[x_fit_full <= current_year]
    y_fit = shifted_exp_func(x_fit - x0, a_fit, b_fit, c_fit)
    regression_line.set_data(x_fit, y_fit)
    regression_line.set_color(interpolate_color(current_year))

    emission_line.set_data(x_data[:frame + 1], emissions[:frame + 1])
    return list(bars) + [emission_line, regression_line]

# Pause after final frame
pause_frames = 125
total_frames = len(x_data) + pause_frames

def extended_update(frame):
    if frame < len(x_data):
        return update(frame)
    else:
        return update(len(x_data) - 1)

# Create animation
ani = animation.FuncAnimation(
    fig,
    extended_update,
    frames=total_frames,
    init_func=init,
    blit=False,
    interval=40,
    repeat=True
)

# Save GIF
plt.draw()
ani.save("co2_vs_temp_dual_axis_darkgrey.gif", writer='pillow', fps=25)
plt.close(fig)
