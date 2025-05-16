import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.ticker import FuncFormatter


def load_energy_data():
    cols = ['Fuel', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    # noinspection PyTypeChecker
    df_renewable = pd.read_excel(
        './DataSet/australian_energy_statistics_2024_table_r.xlsx',
        sheet_name='Consumption by fuel',
        usecols='B:J',
        skiprows=6,
        nrows=15,
        header=None
    )
    df_renewable.columns = cols
    to_drop = ['wood and other b', 'bagasse', 'landfill gas',
               'other biogas', 'ethanol', 'biodiesel', 'other liquid biofuels']
    df_renewable = df_renewable[~df_renewable['Fuel'].isin(to_drop)]
    df_renewable.reset_index(drop=True, inplace=True)
    df_renewable['Fuel-Type'] = 'Renewable'
    # noinspection PyTypeChecker
    df_nonrenewable = pd.read_excel(
        './DataSet/australian_energy_statistics_2024_table_d.xlsx',
        sheet_name='AUS',
        usecols='B:O',
        skiprows=62,
        nrows=8,
        header=None
    )
    df_nonrenewable.drop(columns=df_nonrenewable.columns[[5, 6, 7, 13]], inplace=True)
    df_nonrenewable.columns = [
        'year', 'Black coal', 'Brown coal', 'Coke', 'Coal by-products',
        'Refinery input', 'Petroleum products', 'Natural gas',
        'Town gas', 'Other nonrenewable'
    ]
    df_transposed = df_nonrenewable.set_index('year').T.reset_index()
    df_transposed.rename(columns={'index': 'Fuel'}, inplace=True)
    df_transposed['Fuel-Type'] = 'Nonrenewable'

    df_combined = pd.concat([df_renewable, df_transposed], ignore_index=True)
    column_order = ['Fuel-Type', 'Fuel', '2015-16', '2016-17', '2017-18',
                    '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
    df_consumption = df_combined[column_order]
    print('\n' + "-" * 50)
    print(f"Power Consumption Dataset:\n{df_consumption}")
    print("-" * 50)
    return df_consumption


def summarize_consumption_dataset(df):
    print('\n' + "-" * 50)
    print("Key Characteristics of the Consumption Dataset:")
    print("-" * 50)
    rows, cols = df.shape
    print(f"Dataset Size: {rows} rows × {cols} columns\n")
    print("Variables of Interest:")
    print(f"- Fuel-Type: {df['Fuel-Type'].nunique()} unique values (e.g., {df['Fuel-Type'].unique()[:3]})")
    print(f"- Fuel: {df['Fuel'].nunique()} unique values")
    date_columns = [col for col in df.columns if col not in ['Fuel-Type', 'Fuel']]
    print("\nDate/Time Range:")
    print(f"- From: {min(date_columns)}")
    print(f"- To: {max(date_columns)}")
    print(f"- Total Time Periods: {len(date_columns)} years\n")
    fuel_type_counts = df['Fuel-Type'].value_counts()
    print("Fuel-Type Distribution:")
    for fuel_type, count in fuel_type_counts.items():
        print(f"- {fuel_type}: {count} entries")
    print("\nSample Data:")
    print(df.head(3).to_string(index=False))


def calculate_renewable_percentage(df):
    year_columns = df.columns[2:]
    renewable_df = df[df['Fuel-Type'] == 'Renewable']
    total_by_year = df[year_columns].sum()
    renewable_by_year = renewable_df[year_columns].sum()
    percentage = (renewable_by_year / total_by_year) * 100
    df_renewable_percentages = pd.DataFrame({
        'Year': year_columns,
        'Renewable_Energy_Percentage': percentage.values
    })
    print('\n' + "-" * 50)
    print(f'Percentage of renewable energy consumed in Australia from 2015-16 to 2022-23\n{df_renewable_percentages}')
    print("-" * 50 + '\n')
    return df_renewable_percentages


def predict_renewable_energy(df, target_year):
    df = df.copy()
    df['Year_numeric'] = df['Year'].apply(lambda x: int(x.split('-')[0]) + 0.5)
    X = df['Year_numeric'].values.reshape(-1, 1)
    y = df['Renewable_Energy_Percentage'].values
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_prediction = round(float(linear_model.predict([[target_year + 0.5]])[0]), 4)

    # Polynomial (degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    poly_prediction = round(float(poly_model.predict(poly.transform([[target_year + 0.5]]))[0]), 4)

    print(f"Linear Prediction for {target_year}:\n {linear_prediction}%\n")
    print(f"2nd Order Polynomial Prediction for {target_year}:\n {poly_prediction}%")
    print("-" * 50)
    return linear_prediction, poly_prediction, linear_model, poly_model


def generate_future_labels(start_year, end_year):
    return [f'{y}-{str(y+1)[-2:]}' for y in range(start_year, end_year + 1)]


def plot_renewable_trend(df_percent, lin_model, poly_model, year_linear, value_linear, value_poly):
    df_plot = df_percent.copy()
    df_plot['Year_numeric'] = df_plot['Year'].apply(lambda x: int(x.split('-')[0]) + 0.5)

    plt.figure(figsize=(12, 6))

    # Scatter plot
    plt.scatter(df_plot['Year_numeric'], df_plot['Renewable_Energy_Percentage'],
                color='limegreen', s=80, label='Historical Data', edgecolors='black')

    # Linear fit
    future_years = np.arange(2015.5, 2032.6, 1)
    linear_predictions = lin_model.predict(future_years.reshape(-1, 1))
    plt.plot(future_years, linear_predictions, color='blue', linewidth=2, label='Linear Fit')

    # Polynomial fit
    poly = PolynomialFeatures(degree=2)
    future_poly = poly.fit_transform(future_years.reshape(-1, 1))
    poly_predictions = poly_model.predict(future_poly)
    plt.plot(future_years, poly_predictions, color='blue', linestyle='--', linewidth=2, label='2nd Order Polynomial Fit')

    # Prediction markers
    plt.scatter(year_linear + 0.5, value_linear, color='red', s=200, marker='*', label='Linear Prediction (2030)')
    plt.scatter(year_linear + 0.5, value_poly, color='darkred', s=200, marker='*', label='Polynomial Prediction (2030)')

    # Axes setup
    xticks = np.arange(2015.5, 2032.6, 1)
    xlabels = generate_future_labels(2015, 2032)
    plt.xticks(xticks, xlabels, rotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.xlabel('Year')
    plt.ylabel('Renewable Energy Percentage')
    plt.title('Renewable Energy Trend in Australia (2015–2032)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.savefig('renewable_energy_trend.png', dpi=300)
    plt.show()


def config_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)


def main():
    config_options()
    df_consumption = load_energy_data()
    summarize_consumption_dataset(df_consumption)
    df_renewable_percentages = calculate_renewable_percentage(df_consumption)
    pred_lin, pred_poly, lin_model, poly_model = predict_renewable_energy(df_renewable_percentages, 2030)
    plot_renewable_trend(df_renewable_percentages, lin_model, poly_model, 2030, pred_lin, pred_poly)


if __name__ == "__main__":
    main()
