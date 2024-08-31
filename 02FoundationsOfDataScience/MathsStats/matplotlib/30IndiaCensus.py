import folium
import pandas as pd
import webbrowser
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
india_map = folium.Map(location=[20, 77], zoom_start=4)
indian_cities = folium.map.FeatureGroup()
indian_cities.add_child(folium.CircleMarker(location=[28.7, 77.1], radius=5, color='red', fill=True))
indian_cities.add_child(folium.CircleMarker(location=[19.1, 72.9], radius=5, color='red', fill=True))
indian_cities.add_child(folium.CircleMarker(location=[22.6, 88.4], radius=5, color='red', fill=True))
indian_cities.add_child(folium.CircleMarker(location=[13.1, 80.3], radius=5, color='red', fill=True))
indian_cities.add_child(folium.CircleMarker(location=[13.0, 77.6], radius=5, color='red', fill=True))
indian_cities.add_child(folium.CircleMarker(location=[17.4, 78.5], radius=5, color='red', fill=True))
india_map.add_child(indian_cities)
folium.Marker([28.7, 77.1], popup='New Delhi or any text').add_to(india_map)
folium.Marker([19.1, 72.9], popup='Mumbai: or any text').add_to(india_map)
folium.Marker([22.6, 88.4], popup='Kolkata: or any text').add_to(india_map)
folium.Marker([13.1, 80.3], popup='Chennai: or any text').add_to(india_map)
folium.Marker([13.0, 77.6], popup='Bangalore: or any text').add_to(india_map)
folium.Marker([17.4, 78.5], popup='Hyderabad: or any text').add_to(india_map)

indiaGeojson = r'./data/indiaGeojson.json'

df_census = pd.read_csv('./data/IndiaCensus2011.csv')

folium.Choropleth(
    geo_data=indiaGeojson,
    name="choropleth",
    data=df_census,
    columns=["StateUnionTerritory", "SexRatio"],
    key_on="feature.properties.NAME_1",
    fill_color="YlGn",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Sex Ration of Indian States - 2023",
).add_to(india_map)
folium.LayerControl().add_to(india_map)
india_map.save('./plots/india.html')
webbrowser.open('.\\plots\\india.html', new=2)  # open in new tab
