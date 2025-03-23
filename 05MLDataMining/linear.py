import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

""" Functions for Assessment 2: ZZSC5836 Data Mining and Machine Learning """


def clean_data(df_abalone):
    # Clean dataset: Replace 'M' -> 0, 'F' -> 1, 'I' -> 2 in 'Sex' column
    df_abalone['Sex'] = df_abalone['Sex'].replace({'M': 0, 'F': 1, 'I': 2}).astype(int)
    # One Hot Encoding for Sex Column is Preferred
    # df_abalone = pd.get_dummies(df_abalone, columns=["Sex"], drop_first=True)
    print(f'\n1 # Clean the data (eg. convert M, F and I to 0, 1 and 2).\n{df_abalone.head(5)}')
    return df_abalone


def create_correlation_heatmap(df_data):
    # Develop a correlation map using a heatmap
    corr_matrix = df_data.corr()  # Calculate correlation matrix
    plt.figure(figsize=(9, 9))  # Create a heatmap using seaborn
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
    plt.title('Correlation Heatmap: Abalone Data')
    save_plot('correlation_map')
    plt.show(block=False)


def top_correlated_features(df, target_column='Rings', top_n=2):
    # Identify the two most correlated features with Rings(Age)
    correlation_matrix = df.corr()  # Compute correlation matrix
    target_correlation = correlation_matrix[target_column].drop(target_column)  # Drop self-correlation
    top_features = target_correlation.abs().sort_values(ascending=False).head(top_n).index.tolist()  # Get top N
    print(f'\n3 # Top {top_n} features most correlated with Rings: {top_features}')
    return top_features


def plot_scatter_with_rings(df, top_features):
    # Create scatter plots for the two most correlated features with Rings(Age)
    plt.figure(figsize=(12, 5))
    for i, feature in enumerate(top_features):
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=df[feature], y=df['Rings'], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("Rings (Age)")
        plt.title(f"Scatter Plot: {feature} vs Rings")
    plt.tight_layout()
    save_plot('rings_vs_top_features')
    plt.show(block=False)


def plot_histograms(df, top_features):
    # Create histograms for the two most correlated features with Rings(Age)
    plt.figure(figsize=(12, 4))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df[feature], bins=30, kde=True, color='royalblue')
        plt.title(f'Histogram of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    plt.tight_layout()
    save_plot('histogram')
    plt.show()


def split_data(df):
    """ Splits the dataset into 60% training and 40% testing. """
    random_experiment_number = 21   # Random seed
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=random_experiment_number)
    # print(f"\n5 # Data Set Size: {df.shape}")
    # print(f"\tTraining Set Size: {train_df.shape}")
    # print(f"\tTest Set Size: {test_df.shape}")
    return train_df, test_df


""" Helper functions: Not part of Assessment 2 """


def config_options():
    """ Set options and configurations """
    # set wider console display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    # Explicitly convert string to an integer
    pd.set_option('future.no_silent_downcasting', True)


def load_data(file_path, file_name, column_names):
    # Load the dataset
    return pd.read_csv(file_path + file_name, names=column_names)


def save_plot(filename):
    save_dir = r"./plots"  # Output directory to generate plots/ .png files
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    save_path = os.path.join(save_dir, filename+'.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot


def main():
    """Main function to execute the Linear Model Data Analysis."""
    # set options and configurations:
    config_options()
    # load abalone data file
    file_path = './data/'
    file_name = 'abalone.data'
    column_names = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight",
                    "ShellWeight", "Rings"]
    df_abalone = load_data(file_path, file_name, column_names)
    # Clean abalone data
    df_clean_abalone = clean_data(df_abalone)
    # Generate correlation heatmap
    create_correlation_heatmap(df_clean_abalone)
    # Identify the two most correlated features with Rings(Age)
    top_two_features = top_correlated_features(df_clean_abalone)
    # Create scatter plots for the two most correlated features
    plot_scatter_with_rings(df_clean_abalone, top_two_features)
    # Create Histograms for the two most correlated features
    plot_histograms(df_clean_abalone, top_two_features)
    # Split the dataset into 60% training and 40% testing
    train_data, test_data = split_data(df_clean_abalone)
    print(f"\n5 # Data Set Size: {df_clean_abalone.shape}")
    print(f"\tTraining Set Size: {train_data.shape}")
    print(f"\tTest Set Size: {test_data.shape}")


if __name__ == "__main__":
    main()
