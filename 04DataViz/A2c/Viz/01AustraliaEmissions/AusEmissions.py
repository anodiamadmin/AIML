import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

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
ax1.set_xlim(1989, 2036)
ax1.set_ylim(0, max(co2[~np.isnan(co2)]) * 1.1)
ax2.set_ylim(0, max(gdp[~np.isnan(gdp)]) * 1.1)

ax1.set_ylabel("CO₂ Emissions (Million Tons)", color='dimgray', fontsize=12)
ax2.set_ylabel("GDP (Trillion AU$)", color='red', fontsize=12)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_title("Australia: CO₂ Emissions Target", fontsize=14)

# Background image
fig.figimage(bg_img, xo=0, yo=0, zorder=0, alpha=0.4)

# Line plots
co2_line, = ax1.plot([], [], color='dimgray', linewidth=6, zorder=2, label='CO₂ Emissions')
gdp_line, = ax2.plot([], [], color='red', linewidth=6, zorder=3, label='GDP')

# Tick colors
ax1.tick_params(axis='y', labelcolor='dimgray')
ax2.tick_params(axis='y', labelcolor='red')

# Reference lines and labels (on top)
co2_2005 = df[df['year'] == 2005]['co2'].values[0]
co2_2030_target = round(co2_2005 * 0.57, 2)
annotation_color = 'dimgray'

# Horizontal lines and labels
h1 = ax1.axhline(y=co2_2005, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
t1 = ax1.text(2030, co2_2005 + 5, f"2005 level ({co2_2005:.1f})", color=annotation_color,
              fontsize=10, ha='right', zorder=11)

h2 = ax1.axhline(y=co2_2030_target, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
t2 = ax1.text(2030, co2_2030_target + 5, f"2030 target (43% cut) ({co2_2030_target:.1f})",
              color=annotation_color, fontsize=10, ha='right', zorder=11)

v1 = ax1.axvline(x=2005, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)
v2 = ax1.axvline(x=2030, color=annotation_color, linestyle='dotted', linewidth=2, zorder=10)

# Initialization
def init():
    co2_line.set_data([], [])
    gdp_line.set_data([], [])
    return co2_line, gdp_line, h1, h2, t1, t2, v1, v2

# Animation update
def update(frame):
    delay_frames = 75  # 3 seconds at 25 fps

    if frame < delay_frames:
        # Return empty frame for initial delay
        co2_line.set_data([], [])
        gdp_line.set_data([], [])
        return co2_line, gdp_line, h1, h2, t1, t2, v1, v2
    else:
        # Draw lines normally from this frame onward
        adjusted_frame = frame - delay_frames
        x = years[:adjusted_frame + 1]
        y_co2 = co2[:adjusted_frame + 1]
        y_gdp = gdp[:adjusted_frame + 1]
        co2_line.set_data(x, y_co2)
        gdp_line.set_data(x, y_gdp)
        return co2_line, gdp_line, h1, h2, t1, t2, v1, v2

# Pause at end
pause_frames = 125
total_frames = 75 + len(years) + pause_frames  # 75 = 3 sec pre-delay

# Animate
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, init_func=init,
    blit=False, interval=100, repeat=True
)

# Save animation
ani.save("Australia_GDP_vs_CO2_2030Target.gif", writer='pillow', fps=25)
plt.close(fig)
