import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio.v2 as imageio
import os

# Configuration
csv_file = "CO2PerCapita.csv"
bg_image_file = "Sydney.jpg"
output_gif = "gridEmission.gif"
frames_dir = "frames_emission"
os.makedirs(frames_dir, exist_ok=True)

# Set font
plt.rcParams["font.family"] = "Oxygen"

# Read the data
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# Ensure proper types
df["Year"] = df["Year"].astype(int)
df["co2_per_capita"] = pd.to_numeric(df["co2_per_capita"], errors='coerce')

# Filter for desired countries
countries = ["Australia", "United States", "European Union", "China"]
df = df[df["country"].isin(countries)]

# Load background image
bg_image = mpimg.imread(bg_image_file)

# Get list of years
years = sorted(df["Year"].unique())
frame_paths = []

# Generate frame for each year
for year in years:
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add background image
    ax.imshow(bg_image, extent=[2000, 2024, 0, 25], aspect='auto', alpha=0.3, zorder=0)

    for country in countries:
        subset = df[(df["country"] == country) & (df["Year"] <= year)]
        ax.plot(subset["Year"], subset["co2_per_capita"], label=country, linewidth=2, zorder=2)

    # Axes setup
    ax.set_xlim(2000, 2024)
    ax.set_ylim(0, 25)

    # Titles and labels
    ax.set_title("CO$_2$ Emissions Per Capita (Metric Tons)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Year", fontsize=12, fontweight='bold')
    ax.set_ylabel("CO$_2$ Emissions per Capita (metric tons)", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Make tick labels non-bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    # Save frame
    frame_path = os.path.join(frames_dir, f"frame_{year}.png")
    plt.tight_layout()
    plt.savefig(frame_path, dpi=100)
    plt.close()
    frame_paths.append(frame_path)

# Create gif
images = [imageio.imread(path) for path in frame_paths]
imageio.mimsave(output_gif, images, duration=1.0)

print(f"✅ GIF saved as: {output_gif}")
