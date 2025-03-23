import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score


""" Functions for Assessment 2: ZZSC5836 Data Mining and Machine Learning """


def clean_data(df_abalone):
    # Clean dataset: Replace 'M' -> 0, 'F' -> 1, 'I' -> 2 in 'Sex' column
    # df_abalone['Sex'] = df_abalone['Sex'].replace({'M': 0, 'F': 1, 'I': 2}).astype(int)
    # One Hot Encoding for Sex Column is Preferred
    df_abalone = pd.get_dummies(df_abalone, columns=["Sex"], drop_first=True)
    print(f'\n1 # Clean the data (eg. One Hot Encoding for Sex Column).\n{df_abalone.head(5)}')
    return df_abalone


def split_data(df, target_column):
    """ Splits the dataset into train and test sets (60% train, 40% test) """
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable
    return train_test_split(X, y, test_size=0.40, random_state=42)


def initialize_params(n_features):
    """ Initialize parameters """
    return np.zeros(n_features), 0  # Weights (w) and Bias (b)


# # User defined gradient_descent function: Though the same is used from Scikit Learn Library
# def gradient_descent(X, y, lr=0.01, epochs=1000):
#     """ Gradient Descent Function """
#     m, n = X.shape
#     w, b = initialize_params(n)
#     for _ in range(epochs):
#         # Predictions
#         y_predict = np.dot(X, w) + b
#         # Compute Gradients
#         dw = (-2 / m) * np.dot(X.T, (y - y_predict))
#         db = (-2 / m) * np.sum(y - y_predict)
#         # Update Parameters
#         w -= lr * dw
#         b -= lr * db
#     return w, b


# # User defined mean_squared_error function: Though the same is used from Scikit Learn Library
# def mean_squared_error(y_true, y_hat):
#     """  Compute the Mean Squared Error (MSE) between actual and predicted values  """
#     m = len(y_true)  # Number of samples
#     return np.sum((y_true - y_hat) ** 2) / m


# # User defined r2_score function: Though the same is used from Scikit Learn Library
# def r2_score(y_true, y_hat):
#     """ Compute the R-squared (R²) score """
#     # Total Sum of Squares (TSS)
#     ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
#     # Residual Sum of Squares (RSS)
#     ss_residual = np.sum((y_true - y_hat) ** 2)
#     # R² score
#     r2 = 1 - (ss_residual / ss_total)
#     return r2


def linear_regression(df, normalization_type, show_viz=True):
    """ Performs Linear Regression using Scikit-Learn's Stochastic Gradient Descent. """
    # Split Data
    X_train, X_test, y_train, y_test = split_data(df, "Rings")
    # Normalize Data if required
    if normalization_type == "standard-normalization":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # Train Model using Scikit-Learn SGDRegressor
    model = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_test)
    # Model Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    if show_viz:
        # Visualization: Actual vs. Predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Rings (Age)")
        plt.ylabel("Predicted Rings (Age)")
        plt.title(f"Actual vs Predicted Ring Age: {normalization_type} data")
        save_plot('prediction-' + normalization_type)
        plt.show()
    return rmse, r2


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
    # # linear Regression
    rmse, r2 = linear_regression(df_clean_abalone, 'un-normalized', True)
    # Print Results
    print(f"\n3 # Linear Regression Model Evaluation: un-normalized data"
          f"\n\tRoot Mean Squared Error (R.M.S.E.): {rmse:.2f}\n\tR-squared Score: {r2:.3f}")

    # # normalization of training features
    rmse, r2 = linear_regression(df_clean_abalone, 'standard-normalization', True)
    # Print Results
    print(f"\n4 # Linear Regression Model Evaluation: standard-normalization data"
          f"\n\tRoot Mean Squared Error (R.M.S.E.): {rmse:.2f}\n\tR-squared Score: {r2:.3f}")

    # Repeat 30 times
    rmse_unnormal_list, rmse_normal_list = [], []
    r2_unnormal_list, r2_normal_list = [], []
    num_experiments=30
    for exp_num in range(1, num_experiments + 1):
        rmse, r2 = linear_regression(df_clean_abalone, 'un-normalized', False)
        rmse_unnormal_list.append(rmse)
        r2_unnormal_list.append(r2)
        rmse, r2 = linear_regression(df_clean_abalone, 'standard-normalization', False)
        rmse_normal_list.append(rmse)
        r2_normal_list.append(r2)
    print(rmse_unnormal_list, rmse_normal_list, r2_unnormal_list, r2_normal_list)


if __name__ == "__main__":
    main()
