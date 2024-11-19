import pandas as pd


def load_data(filename):
    """Loads data from a CSV file into a Pandas DataFrame.
    Args:
        filename: The name of the CSV file.
    Returns:
        A Pandas DataFrame containing the data from the file.
    """
    return pd.read_csv(filename)


def find_busiest_routes(df, number_of_routes=10):
    """Finds the required number of (default = 10) the busiest routes based on total passenger trips.
    Args:
        df: The DataFrame containing the flight data.
        number_of_routes: The integer number of busiest routes to return. (Default = 10)
    Returns:
        A Pandas Series with the given number of busiest routes and their total passenger trips.
    """
    busiest_routes = df.groupby(['City1', 'City2'])['PassengerTrips'].sum().nlargest(number_of_routes)
    return busiest_routes


def find_busiest_month_per_route(df):
    """Finds the busiest month for each route based on passenger trips.
    Args:
        df: The DataFrame containing the flight data.
    Returns:
        A Pandas DataFrame with the busiest month number and passenger trips for each route.
    """
    # Group by route and find the month with the maximum passenger trips
    busiest_month_index = df.groupby(['City1', 'City2'])['PassengerTrips'].idxmax()  # required row id's
    busiest_month_num = df.loc[busiest_month_index, 'MonthNum'].values  # busiest month numbers for routes
    busiest_month_trips = df.groupby(['City1', 'City2'])['PassengerTrips'].max()  # number of passenger trips
    busiest_month_df = pd.DataFrame({'BusiestMonth': busiest_month_num, 'NumTrips': busiest_month_trips})
    return busiest_month_df


def find_overall_busiest_month(df):
    """Finds the month with the highest total passenger trips across all routes.
    Args:
        df: The DataFrame containing the flight data.
    Returns:
        A Pandas Series containing the month number with the highest total passenger trips and its corresponding value.
    """
    busiest_month_series = df.groupby('MonthNum')['PassengerTrips'].sum()
    busiest_month_series = busiest_month_series[busiest_month_series == busiest_month_series.max()]
    return busiest_month_series


def find_routes_with_highest_vacancy(df, number_of_routes=10):
    """Finds the required number (default = 10) of routes with the highest proportion of vacant seats.
    Args:
        df: The DataFrame containing the flight data.
        number_of_routes: The integer number of most vacant routes to return. (Default = 10)
    Returns:
        A Pandas dataframe with the given number of most vacant routes and their vacancy proportions.
    """
    # route wise total of available seats
    total_seats_per_route = df.groupby(['City1', 'City2'])['Seats'].sum()
    # route-wise total occupied seats
    total_passenger_trips_per_route = df.groupby(['City1', 'City2'])['PassengerTrips'].sum()
    # route-wise total vacancy/ total available seats
    vacancy_proportion = ((total_seats_per_route - total_passenger_trips_per_route)
                          / total_seats_per_route)
    # top routes with the highest vacancies
    highest_vacancy_routes = vacancy_proportion.nlargest(number_of_routes)
    # Convert Series to DataFrame with desired index [City1', 'City2'] & column name 'Vacancy' as required
    highest_vacancy_routes_df = pd.DataFrame(highest_vacancy_routes, columns=['Vacancy'])
    highest_vacancy_routes_df.index.names = ['City1', 'City2']  # Set index names
    return highest_vacancy_routes_df


def main():
    """Main function to execute the Pandas Data analysis."""
    # Load CSV Data into dataframe
    df = load_data('data.csv')

    # 1. Busiest Routes
    number_of_busiest_routes = 10
    busiest_routes = find_busiest_routes(df, number_of_busiest_routes)
    print(f"\n\nThe {number_of_busiest_routes} busiest routes:\n\n", busiest_routes)

    # 2. Busiest Month per Route
    busiest_month_df = find_busiest_month_per_route(df)
    print("\n\nEach route's busiest month:\n\n", busiest_month_df)
    print("\n\nDisplay all 70 routes as follows:\n\nEach route's busiest month:\n\n",
          busiest_month_df.to_string())

    # 3. Overall Busiest Month
    busiest_month_total = find_overall_busiest_month(df)
    print("\n\nThe overall busiest month:\n\n", busiest_month_total)

    # 4. Routes with Highest Vacancy
    number_of_vacant_routes = 10
    vacant_routes = find_routes_with_highest_vacancy(df, number_of_vacant_routes)
    print(f"\n\nThe {number_of_vacant_routes} routes with the highest proportion of vacant seats:\n\n",
          vacant_routes)


if __name__ == "__main__":
    main()
