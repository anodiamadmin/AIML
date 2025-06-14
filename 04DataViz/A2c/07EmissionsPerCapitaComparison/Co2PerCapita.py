import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("CO2PerCapita.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('\u00A0', '', regex=True)
df.columns = df.columns.str.replace('\u200b', '', regex=True)

# List of countries to compare
countries = ['Australia', 'United States', 'China', 'European Union']

# Plotting
plt.figure(figsize=(12, 6))

for country in countries:
    country_df = df[df['country'] == country]
    plt.plot(country_df['Year'], country_df['co2_per_capita'], marker='o', label=country)

plt.title("CO₂ Emissions Per Capita (2001–2023)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("CO₂ Emissions Per Capita (metric tons)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plt.savefig("CO2PerCapita_Comparison.png", dpi=300)

# Show the plot
plt.show()
