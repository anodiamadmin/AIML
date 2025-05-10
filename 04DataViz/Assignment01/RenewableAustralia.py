import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import plotly.express as px
import imageio.v2 as imageio
import plotly.graph_objects as go


def assign_state_codes(df_state_energy):
    """Adds state_code and removes 'State' column from both dataframes."""
    state_code_map = {'NSW': 1, 'VIC': 2, 'QLD': 3, 'WA': 4, 'SA': 5, 'TAS': 6, 'NT': 7, 'ACT': 8}
    df_state_energy['state_code'] = df_state_energy['State'].map(state_code_map).astype(int)
    return df_state_energy


def load_energy_data(file_path):
    """ Reads renewable and non-renewable energy data from Excel.
        Parameters: file_path (str): Path to the Excel file.
        Calculates: total energy and percentage of renewable energy
        Returns: df_renewable, df_nonrenewable, df_total, df_percent_renewable """
    df_renewable = pd.read_excel(file_path, sheet_name='Renewable')
    df_nonrenewable = pd.read_excel(file_path, sheet_name='NonRenewable')
    df_total = df_renewable.copy()
    df_total.iloc[:, 1:] = df_renewable.iloc[:, 1:] + df_nonrenewable.iloc[:, 1:]
    df_percent_renewable = df_renewable.copy()
    df_percent_renewable.iloc[:, 1:] = df_renewable.iloc[:, 1:] / df_total.iloc[:, 1:] * 100
    df_renewable = assign_state_codes(df_renewable)
    df_nonrenewable = assign_state_codes(df_nonrenewable)
    df_total = assign_state_codes(df_total)
    df_percent_renewable = assign_state_codes(df_percent_renewable)
    return df_renewable, df_nonrenewable, df_total, df_percent_renewable


def config_options():
    """ Set options and configurations """
    pd.set_option('display.max_columns', None)      # set wider console display options
    pd.set_option('display.width', 500)
    gdf = gpd.read_file("australian-states.geojson")
    gdf = gdf.explode(index_parts=False)  # Explode MultiPolygons into individual Polygon rows
    gdf.to_file("australian-states-polygons.geojson", driver="GeoJSON") # Processed new GeoJSON


def draw_choropleth_animated(df_percent_renewable, df_renewable, df_nonrenewable):
    with open('australian-states-polygons.geojson', 'r') as f:
        geojson_data = json.load(f)
    centroids = {
        1: {"lat": -32.0, "lon": 147.0},  # NSW
        2: {"lat": -39.5, "lon": 146.0},  # VIC
        3: {"lat": -21.0, "lon": 143.0},  # QLD
        4: {"lat": -26.0, "lon": 122.0},  # WA
        5: {"lat": -29.0, "lon": 135.0},  # SA
        6: {"lat": -42.0, "lon": 151.0},  # TAS
        7: {"lat": -20.0, "lon": 134.0},  # NT
    }
    years = [col for col in df_percent_renewable.columns if col not in ['state_code', 'State']]
    years.sort()
    years = years[years.index("2008-09"): years.index("2022-23") + 1]
    df_percent = df_percent_renewable.copy()
    df_ren = df_renewable.copy()
    df_nonren = df_nonrenewable.copy()
    df_percent['state_code'] = df_percent['state_code'].astype(int)
    images = []
    for year in years:
        df = df_percent.copy()
        df['capped'] = df[year].clip(lower=1, upper=100)
        df['log_color'] = np.log10(df['capped'])
        fig = px.choropleth(
            df,
            geojson=geojson_data,
            locations='state_code',
            color='log_color',
            featureidkey='properties.STATE_CODE',
            projection='mercator',
            color_continuous_scale=[
                "#8b0000", "#ff0000", "#ff4500", "#ffa500", "#ffd700",
                "#ffff00", "#adff2f", "#7fff00", "#32cd32", "#228b22"
            ],
        )
        # Create annotation labels
        annotations = []
        for _, row in df.iterrows():
            code = row['state_code']
            centroid = centroids.get(code)
            if not centroid:
                continue
            state_name = row['State']
            renewable = df_ren.loc[df_ren['state_code'] == code, year].values[0]
            nonrenewable = df_nonren.loc[df_nonren['state_code'] == code, year].values[0]
            percent = renewable / (renewable + nonrenewable) * 100
            text_lines = [f"<b>{state_name}: {percent:.1f}%</b>"]
            if state_name == "NSW":
                text_lines.append("including ACT")
            text_lines.append(f"♻️{renewable/1000:.1f} ❌{nonrenewable/1000:.1f}")
            annotations.append(go.Scattergeo(
                lon=[centroid["lon"]],
                lat=[centroid["lat"]],
                text="<br>".join(text_lines),
                mode="text",
                showlegend=False,
                textfont=dict(size=11, color="black"),
                hoverinfo='skip'
            ))
        for trace in annotations:
            fig.add_trace(trace)
        tick_vals = np.log10([1, 3.3, 10, 33, 100])
        fig.update_coloraxes(
            colorbar=dict(
                tickvals=tick_vals,
                ticktext=['1%', '3.3%', '10%', '33%', '100%'],
                title=dict(text='Renewable Power Generation Percentage', side='right')
            ),
            cmin=np.log10(1),
            cmax=np.log10(100)
        )
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            projection_scale=1,
            center=dict(lat=-27, lon=133),
            showcountries=False
        )
        fig.update_layout(
            margin=dict(t=120, b=80, l=10, r=10),
            height=1024,
            width=768,
            paper_bgcolor="white"
        )
        fig.add_annotation(
            text="<b>Renewable Power Generation (%)</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.08, showarrow=False,
            font=dict(size=18, color="black"),
            xanchor="center"
        )
        fig.add_annotation(
            text="<b>in Australian States & Territories</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.02, showarrow=False,
            font=dict(size=18, color="black"),
            xanchor="center"
        )
        fig.add_annotation(
            text=f"<b>Financial Year: {year}</b>",
            xref="paper", yref="paper",
            x=0.5, y=0.96, showarrow=False,
            font=dict(size=18, color="red"),
            xanchor="center"
        )
        fig.add_annotation(
            text="Index: ♻️⇒Renewable(TWh) | ❌⇒Non-renewable(TWh)",
            xref="paper", yref="paper",
            x=0.5, y=-0.05, showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center"
        )
        temp_path = f"temp_{year}.png"
        fig.write_image(temp_path, width=768, height=1024)
        time.sleep(0.2)  # Allow time for rendering to disk
        try:
            img = Image.open(temp_path)
            if img.getbbox():  # Check if image has non-zero content
                images.append(imageio.imread(temp_path))
        except Exception as e:
            print(f"Skipping {year} due to rendering error: {e}")
    # Truncate initial blank/black frames if any accidentally got through
    valid_images = []
    for img in images:
        if np.array(img).mean() > 5:  # skip almost-black frames
            valid_images.append(img)
    if not valid_images:
        print("No valid frames found.")
        return
    # Save video
    imageio.mimsave("RenewableAustralia.mp4", valid_images, fps=1)
    print(f"Saved animation with {len(valid_images)} frames.")
    # Skip the first 3 frames (to avoid blank/black screens)
    trimmed_images = images[3:] if len(images) > 3 else images
    # Save to MP4
    imageio.mimsave("RenewableAustralia.mp4", trimmed_images, fps=1)
    # Cleanup
    for f in os.listdir():
        if f.startswith("temp_") and f.endswith(".png"):
            os.remove(f)


def main():
    """Main function to execute the Data Visualization."""
    config_options()    # set console display options and geojson file formats
    file_path = 'RenewableAustralia.xlsx'
    df_renewable, df_nonrenewable, df_total, df_percent_renewable = load_energy_data(file_path)
    print(f"Renewable Energy Produced (GWh):\n{df_renewable.head()}")
    print(f"\nNon-Renewable Energy Produced (GWh):\n{df_nonrenewable.head()}")
    print(f"\nTotal Produced (GWh):\n{df_total.head()}")
    print(f"\nPercentage of Renewable Energy:\n{df_percent_renewable.head()}")
    draw_choropleth_animated(df_percent_renewable, df_renewable, df_nonrenewable)


if __name__ == "__main__":
    main()
