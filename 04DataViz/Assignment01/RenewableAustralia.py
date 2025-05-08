import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
from dash import Dash, dcc, html, Output, Input
import logging
import plotly.graph_objects as go


def assign_state_codes(df_state_energy):
    """Adds state_code and removes 'State' column from both dataframes."""
    state_code_map = {'NSW': 1, 'VIC': 2, 'QLD': 3, 'SA': 4, 'WA': 5, 'TAS': 6, 'NT': 7, 'ACT': 8}
    df_state_energy['state_code'] = df_state_energy['State'].map(state_code_map).astype(int)   # Map state codes
    # df_state_energy.drop(columns='State', inplace=True)    # Drop the original State column
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

    # Static mapping: centroids for the states (approximate lat/lon)
    centroids = {
        1: {"lat": -33.5, "lon": 147.0},   # NSW
        2: {"lat": -37.0, "lon": 145.0},   # VIC
        3: {"lat": -21.0, "lon": 145.0},   # QLD
        4: {"lat": -28.5, "lon": 133.5},   # SA
        5: {"lat": -31.0, "lon": 122.0},   # WA
        6: {"lat": -20.0, "lon": 134.0},   # NT
        7: {"lat": -42.0, "lon": 146.5}    # TAS
    }

    years = [col for col in df_percent_renewable.columns if col not in ['state_code', 'State']]
    years.sort()
    start_idx = years.index("2008-09")
    end_idx = years.index("2022-23")
    years = years[start_idx:end_idx + 1]

    df_percent = df_percent_renewable.copy()
    df_ren = df_renewable.copy()
    df_nonren = df_nonrenewable.copy()
    df_percent['state_code'] = df_percent['state_code'].astype(int)

    app = Dash(__name__)
    app.layout = html.Div([
        html.H1(id='title', style={'textAlign': 'center'}),
        dcc.Graph(id='choropleth'),
        dcc.Interval(id='interval', interval=1000, n_intervals=0, max_intervals=len(years)-1),
        dcc.Store(id='year-index', data=0)
    ])

    @app.callback(
        Output('choropleth', 'figure'),
        Output('title', 'children'),
        Output('year-index', 'data'),
        Input('interval', 'n_intervals'),
        Input('year-index', 'data')
    )
    def update_map(n_intervals, year_idx):
        if year_idx >= len(years):
            year_idx = len(years) - 1
        year = years[year_idx]

        # Prepare main choropleth data
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
            title='',
            color_continuous_scale=[
                "#8b0000", "#ff0000", "#ff4500", "#ffa500", "#ffd700",
                "#ffff00", "#adff2f", "#7fff00", "#32cd32", "#228b22"
            ]
        )

        # Add annotations
        annotations = []
        for idx, row in df.iterrows():
            code = row['state_code']
            centroid = centroids.get(code)
            if not centroid:
                continue

            state_name = row['State']
            renewable = df_ren.loc[df_ren['state_code'] == code, year].values[0]
            nonrenewable = df_nonren.loc[df_nonren['state_code'] == code, year].values[0]

            # Format annotation text
            text_lines = [f"<b>{state_name}</b>"]
            if state_name == "NSW":
                text_lines.append("including ACT")
            text_lines.append(f"Renewable: {renewable:,.0f} GWh")
            text_lines.append(f"Non-renewable: {nonrenewable:,.0f} GWh")

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

        # Update visuals
        tick_vals = np.log10([1, 3.3, 10, 33, 100])
        tick_texts = ['1%', '3.3%', '10%', '33%', '100%']
        fig.update_coloraxes(
            colorbar=dict(
                tickvals=tick_vals,
                ticktext=tick_texts,
                title='Renewable %'
            ),
            cmin=np.log10(1),
            cmax=np.log10(100)
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 20, "l": 0, "b": 0})

        new_year_idx = year_idx + 1 if year_idx + 1 < len(years) else year_idx
        return fig, f'Renewable Energy by State – {year}', new_year_idx

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(debug=False, port=54321)


def main():
    """Main function to execute the Data Visualization."""
    config_options()    # set console display options and geojson file formats
    file_path = 'RenewableAustralia.xlsx'
    df_renewable, df_nonrenewable, df_total, df_percent_renewable = load_energy_data(file_path)
    # Print sample output
    print(f"Renewable Energy Produced (GWh):\n{df_renewable.head()}")
    print(f"\nNon-Renewable Energy Produced (GWh):\n{df_nonrenewable.head()}")
    print(f"\nTotal Produced (GWh):\n{df_total.head()}")
    print(f"\nPercentage of Renewable Energy:\n{df_percent_renewable.head()}")
    # draw_choropleth_log(df_percent_renewable, '2022-23')
    draw_choropleth_animated(df_percent_renewable, df_renewable, df_nonrenewable)


if __name__ == "__main__":
    main()
