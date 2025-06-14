import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

# Load custom font (Oxygen) – Make sure it's available on your system
font_dirs = ['/usr/share/fonts', '/Library/Fonts', 'C:/Windows/Fonts']
oxygen_font = None
for path in font_dirs:
    try:
        oxygen_font = font_manager.FontProperties(fname=os.path.join(path, 'Oxygen-Regular.ttf'))
        break
    except:
        continue
plt.rcParams['font.family'] = 'Oxygen'

# Load data
df = pd.read_excel("gridEmission.xlsx")
years = df['year'].tolist()

# Load background image
bg_img = mpimg.imread("SydneyBG.png")  # Make sure it's a wide image like 1200x600

# Create frames directory
frames_dir = "frames_emission"
os.makedirs(frames_dir, exist_ok=True)

# Set target info
target_year = 2030
target_value = 82

# Create frames
filenames = []
for i in range(len(years)):
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 12 inches * 100 dpi = 1200 px wide

    # Resize and apply background image to match figure canvas
    fig_width, fig_height = fig.get_size_inches() * fig.dpi
    resized_bg = Image.fromarray((bg_img * 255).astype('uint8')).resize((int(fig_width), int(fig_height)))
    fig.figimage(resized_bg, xo=0, yo=0, alpha=1, zorder=0)

    # Data till current year
    df_sub = df.iloc[:i+1]

    # Plot renewable percentage
    ax1.plot(df_sub['year'], df_sub['renewablePercentage'], color='limegreen', linewidth=4, label='Renewable %')
    ax1.set_ylabel('Renewable Percentage (%)', color='green', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.yaxis.label.set_fontweight('bold')
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_xlim(2014, 2031)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add horizontal target line
    ax1.axhline(y=target_value, color='blue', linestyle='dotted', linewidth=2)
    ax1.text(2025.8, target_value + 2, 'Target for 2030: 82% Renewable Energy', fontsize=10, color='blue')

    # Secondary axis for emission intensity
    ax2 = ax1.twinx()
    ax2.plot(df_sub['year'], df_sub['EmissionIntensity'], color='dimgray', linewidth=4, label='Emission Intensity')
    ax2.set_ylabel('Emission Intensity (t/GWh)', color='dimgray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax2.yaxis.label.set_fontweight('bold')
    ax2.set_ylim(0, 1.0)

    # Add vertical line for 2030
    ax1.axvline(x=2030, linestyle='dotted', color='blue', linewidth=2)

    # Draw the dark green dotted trajectory line from 2023 to 2030
    if years[i] >= 2023:
        ax1.plot(
            [2023, 2030],
            [df.loc[df['year'] == 2023, 'renewablePercentage'].values[0], 82],
            color='darkgreen', linestyle='dotted', linewidth=4
        )

    # Title
    plt.title("Australia's Renewable Energy & Grid Emission Trend", fontsize=14, fontweight='bold')

    # Add custom legend/index at bottom right
    custom_lines = [
        Line2D([0], [0], color='dimgray', linewidth=4, label='Emission Intensity'),
        Line2D([0], [0], color='limegreen', linewidth=4, label='Renewable Percentage'),
        Line2D([0], [0], color='darkgreen', linewidth=4, linestyle='dotted', label='Required Trajectory')
    ]
    ax1.legend(handles=custom_lines,
               loc='lower right',
               frameon=False,
               fontsize=9,
               bbox_to_anchor=(1.0, -0.2))

    # Save frame
    fname = os.path.join(frames_dir, f"frame_{i}.png")
    filenames.append(fname)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.15)
    plt.savefig(fname, dpi=100)
    plt.close()

# Generate GIF
with imageio.get_writer("gridEmission.gif", mode='I', duration=1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optional: Clean up frames
for f in filenames:
    os.remove(f)

print("GIF saved as 'gridEmission.gif'")
