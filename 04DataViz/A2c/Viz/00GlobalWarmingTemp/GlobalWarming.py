import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit

# Load Excel data
file_path = "GlobalWarmingAbove20thCentury.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Prepare data
x_data = df['Year'].values
y_data = df['Anomaly'].values
x0 = x_data.min()
x_norm = x_data - x0  # Normalize years for fitting


# Define exponential model: y = a * exp(b * x)
def exp_func(x, a, b):
    return a * np.exp(b * x)


# Fit exponential model
params, _ = curve_fit(exp_func, x_norm, y_data, p0=(0.01, 0.01))
a_fit, b_fit = params
print(f"Exponential fit: y = {a_fit:.4f} * exp({b_fit:.4f} * x)")

# Prepare figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Global Temperature Anomalies Over the Years", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.grid(True)
ax.set_xlim(x_data.min(), x_data.max())
ax.set_ylim(y_data.min() - 0.1, y_data.max() + 0.3)

# Set background split at y=0
ax.axhspan(0, y_data.max() + 0.3, facecolor='#ffdddd', zorder=0)
ax.axhspan(y_data.min() - 0.1, 0, facecolor='#ddeeff', zorder=0)

# Initialize variables
temperature_line = None
regression_line, = ax.plot([], [], 'k--', linewidth=2, label="Exponential Fit")
ax.legend()


# Generate colored temperature segments
def get_colored_line_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['red' if y[i] > 0 else 'blue' for i in range(len(y) - 1)]
    return LineCollection(segments, colors=colors, linewidths=3)


# Init function
def init():
    return []


# Update function
def update(frame):
    # Remove old line segments
    for coll in ax.collections[:]:
        coll.remove()

    # Reapply background spans
    ax.axhspan(0, y_data.max() + 0.3, facecolor='#ffdddd', zorder=0)
    ax.axhspan(y_data.min() - 0.1, 0, facecolor='#ddeeff', zorder=0)

    # Show temperature data until full length
    if frame < len(x_data):
        x = x_data[:frame + 1]
        y = y_data[:frame + 1]
        lc = get_colored_line_segments(x, y)
        ax.add_collection(lc)
        return [lc]

    # After all temp points, animate regression line
    reg_frames = frame - len(x_data)
    reg_x = np.linspace(x_data.min(), x_data.max(), 200)
    reg_y = exp_func(reg_x - x0, a_fit, b_fit)
    regression_line.set_data(reg_x[:reg_frames], reg_y[:reg_frames])
    return [regression_line]


# Total frames = temperature + regression animation (extra 100 frames for smooth regression draw)
total_frames = len(x_data) + 100

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, init_func=init, blit=False, interval=30
)

# Save GIF
ani.save("temperature_anomalies_animation.gif", writer='pillow', fps=30)
plt.close(fig)
