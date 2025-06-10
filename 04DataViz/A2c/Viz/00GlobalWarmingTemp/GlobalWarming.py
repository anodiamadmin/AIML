import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# Load Excel file
file_path = "GlobalWarmingAbove20thCentury.xlsx"  # Ensure the path is correct
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Prepare the figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Global Temperature Anomalies Over the Years", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.grid(True)

# Set fixed axis limits
ax.set_xlim(df['Year'].min(), df['Year'].max())
ax.set_ylim(df['Anomaly'].min() - 0.1, df['Anomaly'].max() + 0.1)

# Draw static background color split at y=0
ax.axhspan(0, df['Anomaly'].max() + 0.1, facecolor='#ffdddd', zorder=0)  # Red above x-axis
ax.axhspan(df['Anomaly'].min() - 0.1, 0, facecolor='#ddeeff', zorder=0)  # Blue below x-axis

# Compute and draw logarithmic regression line
x_data = df['Year'].values
y_data = df['Anomaly'].values
log_x = np.log(x_data - x_data.min() + 1)  # Shift to avoid log(0)
coeffs = np.polyfit(log_x, y_data, deg=1)
log_fit = coeffs[0] * log_x + coeffs[1]
ax.plot(x_data, log_fit, color='black', linestyle='--', linewidth=2, label='Logarithmic Fit')
ax.legend()

# Function to generate colored line segments based on y-value sign
def get_colored_line_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['red' if y[i] > 0 else 'blue' for i in range(len(y) - 1)]
    lc = LineCollection(segments, colors=colors, linewidths=3)
    return lc

# Initialization function
def init():
    return []

# Animation update function
def update(frame):
    x = df['Year'].iloc[:frame + 1].values
    y = df['Anomaly'].iloc[:frame + 1].values

    # Remove old lines
    # while ax.collections:
    #     ax.collections.pop()
    for coll in ax.collections[:]:
        coll.remove()

    # Recreate background spans (since they were removed)
    ax.axhspan(0, df['Anomaly'].max() + 0.1, facecolor='#ffdddd', zorder=0)
    ax.axhspan(df['Anomaly'].min() - 0.1, 0, facecolor='#ddeeff', zorder=0)

    # Create new line segment
    lc = get_colored_line_segments(x, y)
    ax.add_collection(lc)

    return [lc]

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init, blit=False, interval=30
)

# Save the animation as a GIF
ani.save("temperature_anomalies_animation.gif", writer='pillow', fps=30)
plt.close(fig)
