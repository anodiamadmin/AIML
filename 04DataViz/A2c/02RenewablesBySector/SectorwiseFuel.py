import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

def main():
    # File and directory setup
    excel_file = "SectorwiseFuel.xlsx"
    gif_file = "SectorwiseEnergyAnimation.gif"
    frames_dir = "frames_sectorwise_energy"
    os.makedirs(frames_dir, exist_ok=True)

    # Load data
    df = pd.read_excel(excel_file, sheet_name='IndustryVsFuel')
    df.rename(columns={'Fuel': 'Fuel Type'}, inplace=True)

    # Define year columns and desired sector order
    years = df.columns[2:]
    sector_order = [
        "Transport",
        "Primary Industries (Agriculture & Mining)",
        "Manufacturing",
        "Residential",
        "Construction & Other Industries"
    ]

    image_paths = []

    # Create intro frame
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.text(0.5, 0.5, "Sector-wise Renewable vs NonRenewable Energy Use\nAustralia (2003–2023)",
            ha='center', va='center', fontsize=18, fontweight='bold')
    intro_path = os.path.join(frames_dir, "frame_intro.png")
    plt.savefig(intro_path)
    plt.close()
    image_paths.extend([intro_path] * 5)  # Pause on intro for ~5 seconds

    total_frames = len(years)

    for i, year in enumerate(years):
        # Pivot and reorder
        df_year = df[['Industry', 'Fuel Type', year]].copy()
        df_pivot = df_year.pivot(index='Industry', columns='Fuel Type', values=year).fillna(0)
        df_pivot = df_pivot.loc[sector_order]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_pivot[['NonRenewable', 'Renewable']].plot(
            kind='bar',
            stacked=True,
            color=['#a6977f', '#4cff4c'],
            ax=ax
        )

        ax.set_ylim(0, 1850)

        # Titles and Year
        ax.text(0.5, 1.10, "Industry/ Sector - Wise Fuel Consumption (Peta Jules)",
                ha='center', va='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 1.03, f"Financial Year - {year}", color='red',
                ha='center', va='center', fontsize=20, fontweight='bold', transform=ax.transAxes)

        # Axis labels
        ax.set_ylabel("Energy (PJ)", fontweight='bold')
        ax.set_xlabel("Industry/ Sector", fontweight='bold')

        # Custom x-tick labels
        ax.set_xticklabels([
            "Transport",
            "Primary Industries\n(Agriculture & Mining)",
            "Manufacturing",
            "Residential",
            "Construction &\nOther Industries"
        ], rotation=0, ha='center')

        # Simplified bar labels — all white and centered
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f"{height:.0f}",
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                                ha='center', va='center',
                                fontsize=10, fontweight='bold', color='#333333')

        plt.tight_layout()
        frame_path = os.path.join(frames_dir, f"frame_{year}.png")
        plt.savefig(frame_path)
        image_paths.append(frame_path)
        plt.close()

    # Create GIF: intro 5s, each year 2.5s
    images = [imageio.imread(path) for path in image_paths]
    durations = [2.0] * len(images)
    imageio.mimsave(gif_file, images, duration=durations)

    print(f"✅ Animation saved as: {gif_file}")

if __name__ == "__main__":
    main()
