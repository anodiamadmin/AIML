import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.colors import to_rgba

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

# Exponential model: y = c + a * exp(b * x)
def shifted_exp_func(x, a, b, c):
    return c + a * np.exp(b * x)

# Fit temperature curve
params, _ = curve_fit(shifted_exp_func, x_norm, y_data, p0=(0.01, 0.01, -0.2), maxfev=10000)
a_fit, b_fit, c_fit = params

# Generate regression data
x_fit_full = np.linspace(x_data.min(), x_data.max(), 500)
y_fit_full = shifted_exp_func(x_fit_full - x0, a_fit, b_fit, c_fit)

# Color interpolation from 1940 to 1970
def interpolate_color(year):
    start, end = 1940, 1970
    if year <= start:
        return to_rgba('skyblue')
    elif year >= end:
        return to_rgba('red')
    else:
        t = (year - start) / (end - start)
        return tuple((1 - t) * np.array(to_rgba('skyblue')) + t * np.array(to_rgba('red')))

# Create figure and twin axes
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Labels
ax1.set_title("Global CO₂ Emissions vs Temperature Anomalies Over the Years", fontsize=14)
ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature Anomaly (°C)")
ax2.set_ylabel("CO₂ Emissions (Billion Tons)")

# Axis limits
ax1.set_xlim(x_data.min() - 1, x_data.max() + 1)
ax1.set_ylim(min(y_data.min(), y_fit_full.min()) - 0.2, max(y_data.max(), y_fit_full.max()) + 0.2)
ax2.set_ylim(-19.25, 40)  # As requested

# Background color zones (on ax1)
ax1.axhspan(0, y_data.max() + 0.5, facecolor='#ffdddd', zorder=0)
ax1.axhspan(y_data.min() - 0.5, 0, facecolor='#ddeeff', zorder=0)

# Temperature anomaly bars
bars = ax1.bar(x_data, np.zeros_like(y_data), width=1.0, color='grey', zorder=2)

# Regression and Emission lines
regression_line, = ax1.plot([], [], linewidth=6, zorder=4)
emission_line, = ax2.plot([], [], color='darkgrey', linewidth=6, zorder=3)

# Hide emission axis labels/ticks below y = 0
def format_emission_ticks(y, _):
    return f"{y:.0f}" if y > 0 else ''

ax2.set_yticks(np.linspace(-15, 40, 12))  # Add enough ticks
ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_emission_ticks))

# Initialization function
def init():
    for bar in bars:
        bar.set_height(0)
    regression_line.set_data([], [])
    emission_line.set_data([], [])
    return list(bars) + [emission_line, regression_line]

# Update each frame
def update(frame):
    for i in range(frame + 1):
        bars[i].set_height(y_data[i])
        bars[i].set_color('darkred' if y_data[i] >= 0 else 'darkblue')

    # Update regression line
    current_year = x_data[min(frame, len(x_data) - 1)]
    x_fit = x_fit_full[x_fit_full <= current_year]
    y_fit = shifted_exp_func(x_fit - x0, a_fit, b_fit, c_fit)
    regression_line.set_data(x_fit, y_fit)
    regression_line.set_color(interpolate_color(current_year))

    # Update emissions line
    emission_line.set_data(x_data[:frame + 1], emissions[:frame + 1])

    return list(bars) + [emission_line, regression_line]

# Frame logic with 5-second pause
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

# Save animation
plt.draw()
ani.save("co2_vs_temp_dual_axis_darkgrey.gif", writer='pillow', fps=25)
plt.close(fig)
