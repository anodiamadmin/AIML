import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # fixes deprecation warning
import os
import matplotlib.image as mpimg
from matplotlib import rcParams
from PIL import Image

# Set global font
rcParams['font.family'] = 'Oxygen'

def set_background_image(ax, image_path):
    """Set stretched background image behind entire plot."""
    img = mpimg.imread(image_path)
    ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1], transform=ax.transAxes, zorder=0)

def main():
    # File and directory setup
    excel_file = "SectorwiseFuel.xlsx"
    gif_file = "SectorwiseEnergyAnimation.gif"
    frames_dir = "frames_sectorwise_energy"
    bg_image_path = "SydneyBG.png"
    os.makedirs(frames_dir, exist_ok=True)

    # Load data
    df = pd.read_excel(excel_file, sheet_name='IndustryVsFuel')
    df.rename(columns={'Fuel': 'Fuel Type'}, inplace=True)

    # Define years and sector order
    years = df.columns[2:]
    sector_order = [
        "Transport",
        "Primary Industries (Agriculture & Mining)",
        "Manufacturing",
        "Residential",
        "Construction & Other Industries"
    ]

    image_paths = []

    # Loop through each year
    for year in years:
        # Filter and pivot data
        df_year = df[['Industry', 'Fuel Type', year]].copy()
        df_pivot = df_year.pivot(index='Industry', columns='Fuel Type', values=year).fillna(0)
        df_pivot = df_pivot.loc[sector_order]

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        set_background_image(ax, bg_image_path)

        df_pivot[['NonRenewable', 'Renewable']].plot(
            kind='bar',
            stacked=True,
            color=['#a6977f', '#4cff4c'],
            ax=ax,
            zorder=2
        )

        ax.set_ylim(0, 1850)

        # Titles and labels
        ax.text(0.5, 1.10, "Industry/Sector-Wise Fuel Consumption (PetaJoules)",
                ha='center', va='center', fontsize=16, fontweight='bold', transform=ax.transAxes, zorder=3)
        ax.text(0.5, 1.03, f"Financial Year - {year}", color='red',
                ha='center', va='center', fontsize=18, fontweight='normal', transform=ax.transAxes, zorder=3)

        ax.set_ylabel("Energy (PJ)", fontweight='bold')
        ax.set_xlabel("Industry / Sector", fontweight='bold')

        ax.set_xticklabels([
            "Transport",
            "Primary Industries\n(Agriculture & Mining)",
            "Manufacturing",
            "Residential",
            "Construction &\nOther Industries"
        ], rotation=0, ha='center', fontweight='normal')

        # Bar annotations
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{height:.0f}",
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                                ha='center', va='center',
                                fontsize=10, fontweight='normal', color='#333333', zorder=3)

        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{year}.png")
        plt.savefig(frame_path, dpi=100)
        plt.close()
        image_paths.append(frame_path)

    # Create animated GIF with consistent size
    target_size = (1200, 600)  # width × height in pixels

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
        images.append(img)

    images[0].save(
        gif_file,
        save_all=True,
        append_images=images[1:],
        duration=2000,  # milliseconds per frame
        loop=0
    )

    print(f"✅ Animation saved as: {gif_file}")

if __name__ == "__main__":
    main()
