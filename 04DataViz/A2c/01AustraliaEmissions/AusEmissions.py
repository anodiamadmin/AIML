import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from matplotlib import rcParams

# Set global font to Oxygen
rcParams['font.family'] = 'Oxygen'

# Load data
df = pd.read_excel("AustraliaEmissions.xlsx")

# Extend to 2035
future_years = np.arange(2024, 2036)
blank_data = pd.DataFrame({
    'year': future_years,
    'co2': [np.nan] * len(future_years),
    'gdp': [np.nan] * len(future_years)
})
df = pd.concat([df, blank_data], ignore_index=True)

# Extract values
years = df['year'].values
co2 = df['co2'].values
gdp = df['gdp'].values

# Load background image
bg_img = mpimg.imread("Sydney.jpg")

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Set axis limits and labels
ax1.set_xlim(2004, 2036)
ax1.set_ylim(0, max(co2[~np.isnan(co2)]) * 1.1)
ax2.set_ylim(0, max(gdp[~np.isnan(gdp)]) * 1.3)

# Set axis labels and title (bold)
ax1.set_ylabel("CO$_2$ Emissions (Million Tons)", color='dimgray', fontsize=12, fontweight='bold')
ax2.set_ylabel("GDP (Billion AU$)", color='#ff5c5c', fontsize=12, fontweight='bold')
ax1.set_xlabel("Year", fontsize=12, fontweight='bold')
ax1.set_title("Australia: CO$_2$ Emissions Target", fontsize=14, fontweight='bold')

# Background image
fig.figimage(bg_img, xo=0, yo=0, zorder=0, alpha=0.4)

# Line plots
co2_line, = ax1.plot([], [], color='dimgray', linewidth=6, zorder=2, label='CO₂ Emissions')
gdp_line, = ax2.plot([], [], color='#ff5c5c', linewidth=3, zorder=3, label='GDP')
projection_line, = ax1.plot([], [], linestyle='dotted', color='black', linewidth=2, zorder=5)

# Tick label colors (normal weight)
ax1.tick_params(axis='y', labelcolor='dimgray', labelsize=10)
ax2.tick_params(axis='y', labelcolor='#ff5c5c', labelsize=10)
ax1.tick_params(axis='x', labelsize=10)

# Reference lines and labels
co2_2005 = df[df['year'] == 2005]['co2'].values[0]
co2_2030_target = round(co2_2005 * 0.57, 2)
annotation_color = 'dimgray'

# Horizontal & vertical reference lines
h1 = ax1.axhline(y=co2_2005, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
t1 = ax1.text(
    2030, co2_2005 + 5,
    f"2005 Emission Level ({co2_2005:.1f})",
    color=annotation_color, fontsize=10, fontweight='normal', ha='right', zorder=11
)
h2 = ax1.axhline(y=co2_2030_target, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
t2 = ax1.text(
    2030, co2_2030_target + 5,
    f"2030 Target Level (43% cut) ({co2_2030_target:.1f})",
    color=annotation_color, fontsize=10, fontweight='normal', ha='right', zorder=11
)
v1 = ax1.axvline(x=2005, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
v2 = ax1.axvline(x=2030, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)

# Pause & animation settings
delay_frames = 75  # 3-second pre-animation pause
draw_frames = len(years)
pause_frames = 125  # 5-second end pause
total_frames = delay_frames + draw_frames + pause_frames

# Projection coordinates
proj_x = [2024, 2030]
proj_y = [444.7269224, 359.7]

# Initialization
def init():
    co2_line.set_data([], [])
    gdp_line.set_data([], [])
    projection_line.set_data([], [])
    return co2_line, gdp_line, projection_line, h1, h2, t1, t2, v1, v2

# Animation update
def update(frame):
    if frame < delay_frames:
        # Pre-animation blank
        co2_line.set_data([], [])
        gdp_line.set_data([], [])
        projection_line.set_data([], [])
    elif delay_frames <= frame < delay_frames + draw_frames:
        # Draw lines gradually
        idx = frame - delay_frames
        x = years[:idx + 1]
        co2_line.set_data(x, co2[:idx + 1])
        gdp_line.set_data(x, gdp[:idx + 1])
        projection_line.set_data([], [])  # Hide projection until full data drawn
    else:
        # After full lines drawn, show projection line
        co2_line.set_data(years, co2)
        gdp_line.set_data(years, gdp)
        projection_line.set_data(proj_x, proj_y)

    return co2_line, gdp_line, projection_line, h1, h2, t1, t2, v1, v2

# Animate
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, init_func=init,
    blit=False, interval=100, repeat=True
)

# Save animation
ani.save("Australia_GDP_vs_CO2_2030Target.gif", writer='pillow', fps=25)
plt.close(fig)
