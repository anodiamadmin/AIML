import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter
from PIL import Image
import matplotlib.font_manager as fm
import numpy as np

# Load data
df_numbers = pd.read_csv("RooftopNumbers.csv")
df_capacity = pd.read_csv("RooftopCapacity.csv")

# Merge datasets on Year
df = pd.merge(df_numbers, df_capacity, on="Year")
df = df.sort_values("Year").reset_index(drop=True)

# Load background image with alpha
bg_image = Image.open("Rooftop.jpg").convert("RGBA")
bg_alpha = 0.3
bg_array = np.array(bg_image).astype(float)
bg_array[..., :3] = bg_array[..., :3] * bg_alpha + 255 * (1 - bg_alpha)
bg_array = bg_array.astype(np.uint8)

# Font setup
font_prop_bold = fm.FontProperties(family="Arial", weight="bold", size=12)
font_prop_normal = fm.FontProperties(family="Arial", weight="normal", size=10)

# Prepare figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot setup
years = df["Year"]
installs = df["NumberOfInstallations"]
capacity = df["InstalledCapacityGW"]

# Set axis limits
ax1.set_xlim(2013, 2025)
ax1.set_ylim(0, installs.max() * 1.1)
ax2.set_ylim(0, capacity.max() * 1.1)

# Set background image inside axes
bg_extent = [2013, 2025, 0, installs.max() * 1.1]
ax1.imshow(bg_array, aspect='auto', extent=bg_extent, zorder=0)

# Labels and formatting
ax1.set_xlabel("Year", fontproperties=font_prop_bold)
ax1.set_ylabel("Approximate Number of Installations", fontproperties=font_prop_bold)
ax2.set_ylabel("Installed PV Capacity (GW)", fontproperties=font_prop_bold)
plt.title("Rooftop Solar Installations in Australia (2014–2024)", fontproperties=font_prop_bold)

ax1.tick_params(axis='x', labelsize=9)
ax1.tick_params(axis='y', labelsize=9)
ax2.tick_params(axis='y', labelsize=9)

ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

# Line placeholders
line1, = ax1.plot([], [], color="tab:blue", label="Installations", zorder=2)
line2, = ax2.plot([], [], color="tab:green", label="Capacity (GW)", zorder=2)
text_year = ax1.text(0.05, 0.95, "", transform=ax1.transAxes,
                     fontproperties=font_prop_bold, fontsize=12, verticalalignment='top')

# Init function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    text_year.set_text("")
    return line1, line2, text_year

# Update function
def update(frame):
    current_years = years[:frame + 1]
    current_installs = installs[:frame + 1]
    current_capacity = capacity[:frame + 1]

    line1.set_data(current_years, current_installs)
    line2.set_data(current_years, current_capacity)
    text_year.set_text(f"Year: {years[frame]}")
    return line1, line2, text_year

# Animate
ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init,
    blit=True, interval=1000, repeat=False
)

# Save animation
ani.save("Rooftop.gif", writer="pillow")
plt.close()
