import pandas as pd
import matplotlib.pyplot as plt
import os


def save_plot(plot, filename):
    # Check if the directory exists, if not, create it
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Save the plot
    plot.savefig(f'plots/{filename}.png')
    plot.close()
    print(f"Plot saved as 'plots/{filename}.png'")


# 1. Pie plot for mission outcomes
def plot_outcome_distribution(data):
    outcome_counts = data['Outcome'].value_counts()  # Count occurrences of each outcome
    plt.figure(figsize=(8, 6))
    outcome_counts.plot(kind='pie')
    plt.ylabel('Outcome')
    plt.show()
    save_plot(plt, 'outcome_distribution')


def improved_plot_outcome_distribution(data):
    outcome_counts = data['Outcome'].value_counts()  # Count occurrences of each outcome
    plt.figure(figsize=(8, 6))
    # Explode slices with less than 10% of total missions
    explode = [0.25 if percentage < 10 else 0 for percentage in outcome_counts.values / outcome_counts.sum() * 100]
    colors = ['#44ff44', 'red', 'yellow', 'orange']  # Define colors for each slice
    outcome_counts.plot(
        kind='pie',
        colors=colors,
        explode=explode,
        # Format percentage labels, show only the significant ones > 2%
        autopct=lambda pct: f'{pct:.1f}%' if pct >= 2 else '',
        # Format legend labels
        labels=[f'{outcome} ({percentage:.1f}%)' if
                percentage < 2 else outcome for outcome, percentage in
                zip(outcome_counts.index, outcome_counts.values / outcome_counts.sum() * 100)],
        # start from 1:00 'o clock position
        startangle=60,
        # fontsize=8,
        textprops={'fontsize': 8}
    )
    plt.ylabel('Mission Outcomes Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    save_plot(plt, 'improved_outcome_distribution')


# 2. Horizontal bar plot for total missions by company
def plot_total_missions_by_company(data):
    company_counts = data['Company'].value_counts()  # Count missions by company
    plt.figure(figsize=(10, 8))
    company_counts.sort_values().plot(kind='barh')
    plt.ylabel('Company')
    plt.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.show()
    save_plot(plt, 'total_missions_by_company')


def improved_plot_total_missions_by_company(data):
    company_counts = data['Company'].value_counts()  # Count missions by company
    plt.figure(figsize=(10, 8))
    ax = company_counts.sort_values().plot(kind='barh')  # Get the axes object to set xscale
    ax.set_xscale('log')  # Set x-axis to the logarithmic scale
    # Set xticks to 1, 10, 100, 1000...
    ax.set_xticks([1, 10, 100, 1000, 10000])  # Adjust range as needed
    # Set xtick labels to integer values
    ax.set_xticklabels([1, 10, 100, 1000, 10000])  # Adjust range as needed
    plt.ylabel('Company')
    plt.xlabel('Number of Missions (log scale)')
    plt.tick_params(axis='y', labelsize=7)
    plt.title('Space Missions per Company', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    save_plot(plt, 'improved_total_missions_by_company')


# 3. Horizontal bar plot for missions grouped by outcome (success/failure)
def plot_missions_by_company_and_outcome(data):
    # Group data by company and success/failure
    grouped_data = data.copy()
    grouped_data['Outcome'] = grouped_data['Outcome'].apply(
        lambda x: 'Success' if x in ['Success'] else 'Failure')
    grouped_counts = grouped_data.groupby(['Company', 'Outcome']).size().unstack(fill_value=0)
    # Calculate total missions per company
    grouped_counts['Total'] = grouped_counts['Success'] + grouped_counts['Failure']
    # Sort by total missions
    grouped_counts = grouped_counts.sort_values(by=['Total', 'Success'], ascending=[True, True])
    # Create a horizontal bar plot
    grouped_counts[['Failure', 'Success']].plot(  # Removed Total from plot
        kind='barh',
        stacked=True,
        color={'Success': 'green', 'Failure': 'red'},
        figsize=(10, 8)
    )
    plt.ylabel('Company')
    plt.tick_params(axis='y', labelsize=7)
    plt.legend(title='Outcome', title_fontsize=7, fontsize=7)
    plt.tight_layout()
    plt.show()
    save_plot(plt, 'missions_by_company_and_outcome')


def improved_plot_missions_by_company_and_outcome(data):
    # Group data by company and success/failure
    grouped_data = data.copy()
    grouped_data['Outcome'] = grouped_data['Outcome'].apply(
        lambda x: 'Success' if x in ['Success'] else 'Failure')
    grouped_counts = grouped_data.groupby(['Company', 'Outcome']).size().unstack(fill_value=0)
    # Calculate total missions per company
    grouped_counts['Total'] = grouped_counts['Success'] + grouped_counts['Failure']
    # Sort by total missions
    grouped_counts = grouped_counts.sort_values(by=['Total', 'Success'], ascending=[True, True])
    # Prepare data for y-tick labels with success/failure counts
    y_ticklabels = []
    for index, row in grouped_counts.iterrows():
        label = f"{index} - Failure {row['Failure']} : Success {row['Success']}"
        y_ticklabels.append(label)
    # Create a horizontal bar plot
    ax = grouped_counts[['Failure', 'Success']].plot(  # Removed Total from plot
        kind='barh',
        stacked=True,
        color={'Success': '#22dd22', 'Failure': 'red'},
        figsize=(10, 8)
    )
    plt.yticks(range(len(y_ticklabels)), y_ticklabels)
    plt.ylabel('Company')
    plt.tick_params(axis='y', labelsize=7)
    plt.xlabel('Number of Missions')
    plt.title('Outcome of Space Missions by Companies', fontsize=14, fontweight='bold')
    plt.legend(title='Outcome', title_fontsize=7, fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.show()
    save_plot(plt, 'improved_missions_by_company_and_outcome')


def main():
    """Main function to perform the Data analysis and plot the graphs."""
    # Read the data from the CSV file
    file_path = "data.csv"  # Update the path if necessary
    try:
        data = pd.read_csv(file_path)
        # Ensure columns of interest are properly formatted
        data['Outcome'] = data['Outcome'].str.strip()  # Remove extra whitespace
        data['Company'] = data['Company'].str.strip()

        # Generate the plots exactly as required
        plot_outcome_distribution(data)  # Plot #1
        plot_total_missions_by_company(data)  # Plot #2
        plot_missions_by_company_and_outcome(data)  # Plot #3

        # Generate the improved plots
        improved_plot_outcome_distribution(data)  # Improved Plot #1
        improved_plot_total_missions_by_company(data)  # Improved Plot #2
        improved_plot_missions_by_company_and_outcome(data)  # Improved Plot #3

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
    except KeyError as e:
        print(f"Error: Column '{e.args[0]}' not found in the CSV file.")
    except ValueError as e:
        print(f"Error: {e}")  # Catch the value error for missing columns
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
