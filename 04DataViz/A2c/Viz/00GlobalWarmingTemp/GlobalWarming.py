# Re-import libraries for animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# Load Excel file
file_path = "GlobalWarmingAbove2othCentury.xlsx"  # Make sure this path is correct
df = pd.read_excel(file_path, sheet_name="Sheet1")


# Prepare figure and axis
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Global Temperature Anomalies Over the Years", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature Anomaly (°C)")
ax.grid(True)

# Initialize a line object
line, = ax.plot([], [], lw=2)

# Function to initialize the background of the animation
def init():
    ax.set_xlim(df['Year'].min(), df['Year'].max())
    ax.set_ylim(df['Anomaly'].min() - 0.1, df['Anomaly'].max() + 0.1)
    line.set_data([], [])
    return line,

# Animation function that updates the line
def update(frame):
    x = df['Year'][:frame]
    y = df['Anomaly'][:frame]
    line.set_data(x, y)
    return line,

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init, blit=True, interval=30
)

# Save the animation to a file
output_path = "temperature_anomalies_animation.mp4"
ani.save("temperature_anomalies_animation.gif", writer='pillow', fps=30)
