import pandas as pd
from fontTools.ttLib.tables.otTraverse import dfs_base_table
import geopandas as gpd


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
    return df_renewable, df_nonrenewable, df_total, df_percent_renewable


def config_options():
    """ Set options and configurations """
    pd.set_option('display.max_columns', None)      # set wider console display options
    pd.set_option('display.width', 500)
    gdf = gpd.read_file("australian-states.geojson")
    gdf = gdf.explode(index_parts=False)  # Explode MultiPolygons into individual Polygon rows
    gdf.to_file("australian-states-polygons.geojson", driver="GeoJSON") # Processed new GeoJSON


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


if __name__ == "__main__":
    main()
