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

# Exponential regression function
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Fit regression
params, _ = curve_fit(exp_func, x_norm, y_data, p0=(0.01, 0.01), maxfev=10000)
a_fit, b_fit = params

# Regression data points
x_fit_full = np.linspace(x_data.min(), x_data.max(), 500)
y_fit_full = exp_func(x_fit_full - x0, a_fit, b_fit)

# Color interpolation between sky blue and bright red from 1940 to 1970
def interpolate_color(year):
    start_year, end_year = 1940, 1970
    if year <= start_year:
        return to_rgba('skyblue')
    elif year >= end_year:
        return to_rgba('red')
    else:
        ratio = (year - start_year) / (end_year - start_year)
        blue = np.array(to_rgba('skyblue'))
        red = np.array(to_rgba('red'))
        return tuple((1 - ratio) * blue + ratio * red)

# Set up plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Global Temperature Anomalies Over the Years", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.set_xlim(x_data.min() - 1, x_data.max() + 1)
ax.set_ylim(min(y_data.min(), y_fit_full.min()) - 0.2, max(y_data.max(), y_fit_full.max()) + 0.2)
ax.grid(True)

# Background coloring
ax.axhspan(0, y_data.max() + 0.5, facecolor='#ffdddd', zorder=0)
ax.axhspan(y_data.min() - 0.5, 0, facecolor='#ddeeff', zorder=0)

# Bar containers
bars = ax.bar(x_data, np.zeros_like(y_data), width=1.0, color='grey', zorder=2)

# Regression line and animated circle as arrow head
regression_line, = ax.plot([], [], linewidth=6, zorder=4)  # Thicker line
arrow_head = plt.Circle((0, 0), radius=0.5, color='skyblue', zorder=5)
ax.add_patch(arrow_head)

# Init function
def init():
    for bar in bars:
        bar.set_height(0)
    regression_line.set_data([], [])
    arrow_head.set_radius(0.5)
    arrow_head.set_visible(False)
    return list(bars) + [regression_line, arrow_head]

# Update animation frame
def update(frame):
    # Update bars
    for i in range(frame + 1):
        height = y_data[i]
        bars[i].set_height(height)
        bars[i].set_color('darkred' if height >= 0 else 'darkblue')

    # Regression segment
    current_year = x_data[min(frame, len(x_data) - 1)]
    x_fit = x_fit_full[x_fit_full <= current_year]
    y_fit = exp_func(x_fit - x0, a_fit, b_fit)
    regression_line.set_data(x_fit, y_fit)

    # Arrowhead update
    if len(x_fit) > 0:
        head_x, head_y = x_fit[-1], y_fit[-1]
        arrow_head.center = (head_x, head_y)
        arrow_head.set_visible(True)
        arrow_head.set_radius(0.7)  # Slightly bigger than line width
        arrow_head.set_color(interpolate_color(head_x))
    else:
        arrow_head.set_visible(False)

    # Line color changes to match arrow head
    regression_line.set_color(interpolate_color(current_year))

    return list(bars) + [regression_line, arrow_head]

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(x_data), init_func=init, blit=False, interval=40
)

# Save animation
plt.draw()
ani.save("temperature_anomalies_bars_with_arrow.gif", writer='pillow', fps=30)
plt.close(fig)
