import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
file_path = "data.csv"  # Update the path if necessary
data = pd.read_csv(file_path)

# Ensure columns of interest are properly formatted
data['Outcome'] = data['Outcome'].str.strip()  # Remove extra whitespace
data['Company'] = data['Company'].str.strip()


# 1. Pie plot for mission outcomes
def plot_outcome_distribution(data):
    outcome_counts = data['Outcome'].value_counts()  # Count occurrences of each outcome
    plt.figure(figsize=(8, 6))
    outcome_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                        colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
    plt.title('Mission Outcome Distribution')
    plt.ylabel('')  # Remove default ylabel for cleaner plot
    plt.show()


# 2. Horizontal bar plot for total missions by company
def plot_total_missions_by_company(data):
    company_counts = data['Company'].value_counts()  # Count missions by company
    plt.figure(figsize=(10, 8))
    company_counts.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Total Missions by Company')
    plt.xlabel('Number of Missions')
    plt.ylabel('Company')
    plt.tight_layout()
    plt.show()


# 3. Horizontal bar plot for missions grouped by outcome (success/failure)
def plot_missions_by_company_and_outcome(data):
    # Define success and failure categories
    success_outcomes = ['Success']
    failure_outcomes = ['Prelaunch Failure', 'Partial Failure', 'Failure']

    # Group data by company and success/failure
    grouped_data = data.copy()
    grouped_data['Category'] = grouped_data['Outcome'].apply(
        lambda x: 'Success' if x in success_outcomes else 'Failure')
    grouped_counts = grouped_data.groupby(['Company', 'Category']).size().unstack(fill_value=0)

    # Create a horizontal bar plot
    plt.figure(figsize=(12, 8))
    grouped_counts.sort_values(by='Success', ascending=True).plot(
        kind='barh',
        stacked=True,
        color={'Success': 'green', 'Failure': 'red'},
        figsize=(10, 8)
    )
    plt.title('Missions by Company and Outcome')
    plt.xlabel('Number of Missions')
    plt.ylabel('Company')
    plt.tight_layout()
    plt.show()


# Generate the plots
plot_outcome_distribution(data)  # Image1
plot_total_missions_by_company(data)  # Image2
plot_missions_by_company_and_outcome(data)  # Image3
