import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


# --- Data Wrangling Function ---
def wrangle_abalone_data(file_path):
    """
    Reads, cleans, one-hot encodes, and transforms the abalone data.
    """
    try:
        df = pd.read_csv(file_path)
        df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True, dtype=float)

        bins = [0, 7, 10, 15, np.inf]
        labels = [0, 1, 2, 3]
        df_encoded['AgeClass'] = pd.cut(df_encoded['Rings'], bins=bins, labels=labels, right=True, include_lowest=True)
        df_encoded['AgeClass'] = df_encoded['AgeClass'].astype(int)

        df_clean_data = df_encoded.drop(columns=['Rings'])
        return df_clean_data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during data wrangling: {e}")
        return None


# --- Cost Function ---
def calculate_cost(X, y, theta):
    """
    Computes the cost function for linear regression.
    """
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# --- Modified Gradient Descent Function (returns cost_history and theta_history) ---
def gradient_descent(X, y, theta, alpha, iterations):
    """
    Performs gradient descent to optimize the parameters of linear regression.
    Collects the theta vector and cost at each iteration.
    """
    m = len(y)
    cost_history = []
    current_cost = calculate_cost(X, y, theta)
    cost_history.append(current_cost)
    theta_history_list = []
    theta_history_list.append(theta.flatten())
    for i in range(iterations):
        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient

        # Collect current cost and theta for history
        current_cost = calculate_cost(X, y, theta)
        cost_history.append(current_cost)
        theta_history_list.append(theta.flatten())  # Store flattened theta for easier stacking

    theta_history = np.array(theta_history_list).T  # Transpose to get (n_parameters, n_iterations)

    return theta, theta_history, cost_history


# --- Visualization Function for 3D Surface with Path ---
def plot_cost_3d_surface(X_data, y_data, theta_optimized, theta_history, bias_range, length_range,
                         filename='cost_surface_3d_with_path.png'):
    """
    Generates and saves a 3D plot of the cost function surface against
    Bias and Length parameters, overlaying the gradient descent path.

    Args:
        X_data (np.ndarray): The feature matrix (X_intercept) used in training.
        y_data (np.ndarray): The target vector (y) used in training.
        theta_optimized (np.ndarray): The final optimized theta vector.
        theta_history (np.ndarray): A 2D array of shape (n_parameters, n_iterations)
                                    containing theta values at each iteration.
        bias_range (tuple): A tuple (min_bias, max_bias) for the bias axis.
        length_range (tuple): A tuple (min_length, max_length) for the length axis.
        filename (str): The name of the file to save the plot.
    """
    if not isinstance(X_data, np.ndarray) or not isinstance(y_data, np.ndarray) or \
            not isinstance(theta_optimized, np.ndarray) or not isinstance(theta_history, np.ndarray):
        print("Error: X_data, y_data, theta_optimized, and theta_history must be NumPy arrays.")
        return

    # Ensure there are at least two parameters (Bias and Length) to plot
    if theta_optimized.shape[0] < 2 or theta_history.shape[0] < 2:
        print("Error: Not enough parameters in theta_optimized or theta_history to plot Bias and Length.")
        return

    # --- Generate the Cost Surface Data ---
    num_points = 100  # Resolution of the grid for the surface
    bias_values = np.linspace(bias_range[0], bias_range[1], num_points)
    length_values = np.linspace(length_range[0], length_range[1], num_points)
    B, L = np.meshgrid(bias_values, length_values)  # Create 2D grids for Bias and Length

    Z_cost = np.zeros(B.shape)  # Initialize array to store cost values for the surface

    # Create a base theta vector from optimized values.
    # We will modify only the bias (index 0) and length (index 1) parameters for the surface calculation.
    # The other parameters remain fixed at their final optimized values from training.
    theta_base = theta_optimized.copy()

    # Loop through the meshgrid to calculate the cost for each (Bias, Length) combination
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            theta_temp = theta_base.copy()  # Start with the base (optimized other params)
            theta_temp[0, 0] = B[i, j]  # Set the current bias value for this grid point
            theta_temp[1, 0] = L[i, j]  # Set the current length parameter value for this grid point

            # Calculate the cost with these temporary theta values
            Z_cost[i, j] = calculate_cost(X_data, y_data, theta_temp)

    # --- Create the 3D Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cost surface
    surf = ax.plot_surface(B, L, Z_cost, cmap='viridis', edgecolor='none', alpha=0.7)

    # Set labels and title for the surface plot
    ax.set_xlabel('Bias Term')
    ax.set_ylabel('Length Parameter')
    ax.set_zlabel('Cost Function')
    ax.set_title(f'Cost Function Surface with Gradient Descent Path')

    # Add a color bar for the cost values on the surface
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Cost Value')

    # --- Overlay the Gradient Descent Path ---
    # Extract the bias and length parameters' values over iterations
    bias_path = theta_history[0, :]  # First row of theta_history is Bias
    length_path = theta_history[1, :]  # Second row is Length

    # Calculate the cost at each point along the gradient descent path
    # We need to re-calculate costs using the specific theta values at each step
    # because cost_history itself just tracks the overall cost, not just
    # how it relates to Bias and Length while other params are fixed.
    cost_path = [calculate_cost(X_data, y_data, theta_history[:, i].reshape(-1, 1))
                 for i in range(theta_history.shape[1])]

    # Plot the path as a red line with markers
    ax.plot(bias_path, length_path, cost_path,
            color='red', marker='o', markersize=3,
            label='Gradient Descent Path', linewidth=2)

    # --- Mark the Final Optimized Point ---
    # Get the Bias and Length values from the optimized theta
    optimized_bias = theta_optimized[0, 0]
    optimized_length = theta_optimized[1, 0]
    # Get the final cost at the optimized point
    final_optimized_cost = calculate_cost(X_data, y_data, theta_optimized)

    # Plot the optimized point as a green marker
    ax.scatter(optimized_bias, optimized_length, final_optimized_cost,
               color='green', marker='o', s=100, label='Optimized Point', depthshade=False, edgecolors='black')
    ax.view_init(elev=0, azim=45)

    # Add a legend to distinguish the path and optimized point
    ax.legend()

    # --- Save and Show Plot ---
    plt.savefig(filename)
    print(f"3D cost surface plot with gradient descent path saved as {filename}")
    plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    # Set Console Config for better print output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    np.set_printoptions(threshold=np.inf, linewidth=2000)

    # --- Data Loading and Wrangling ---
    file_path = 'abalone.csv'  # Make sure this file is in the same directory!
    df_clean_abalone = wrangle_abalone_data(file_path)

    if df_clean_abalone is None:
        print("Data loading failed. Please ensure 'abalone.csv' is in the correct path.")
        exit()  # Exit the script if data cannot be loaded

    print(f"\n1# df_clean_abalone.head(4):\n", df_clean_abalone.head(4))

    # --- Prepare X and y for Gradient Descent ---
    y = df_clean_abalone['AgeClass'].values.reshape(-1, 1)
    X = df_clean_abalone.drop(columns=['AgeClass'])

    # Add a column of ones to X for the intercept term (Bias)
    X_intercept = np.hstack((np.ones((X.shape[0], 1)), X.values))

    # --- Set Initial Parameters (theta initialization) ---
    num_features_excluding_intercept = X.shape[1]  # Number of original features
    # Calculate limit for uniform distribution for feature weights (He/Xavier-like initialization)
    limit = np.sqrt(6 / num_features_excluding_intercept)
    feature_weights = np.random.uniform(low=-limit, high=limit, size=(num_features_excluding_intercept, 1))
    bias_term = np.zeros((1, 1))  # Initialize bias to zero

    # theta includes bias as the first element and then feature weights
    theta = np.vstack((bias_term, feature_weights))

    print("\n2# Initial Theta (first element is bias, others are feature weights):\n", theta.T)

    # --- Set Hyperparameters ---
    alpha = 0.2
    iterations = 40

    start_time = time.perf_counter()  # Start timing gradient descent

    # --- Perform Gradient Descent ---
    theta_optimized, theta_history, cost_history = gradient_descent(X_intercept, y, theta, alpha, iterations)

    end_time = time.perf_counter()  # End timing

    # --- Print Results ---
    print(f"\n3# Optimized Theta:\n{theta_optimized.T}")
    print(f"\n4# Theta History Shape = {theta_history.shape}\nTheta History:\n{theta_history}")
    print(f"\n5# Cost History Length = {len(cost_history)}\nCost History Length\n{cost_history}")
    print(f"\n6# Time Required = {end_time - start_time:.4f} seconds")

    final_cost = calculate_cost(X_intercept, y, theta_optimized)
    print(f"\n7# Final Cost after optimization = {final_cost:.4f}")

    # # --- Define Parameter Names for 2D Plots (for reference, not used in this specific 3D plot function) ---
    # parameter_names_2d = [
    #     'Bias Term', 'Length', 'Diameter', 'Height', 'WholeWeight',
    #     'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Sex_I', 'Sex_M'
    # ]
    #
    # # --- Call the 3D Plotting Function with Gradient Descent Path ---
    # # Define the ranges for Bias and Length as requested
    # bias_plot_range = (-0.1, 0.7)
    # length_plot_range = (-0.2, 0.6)
    #
    # plot_cost_3d_surface(X_intercept, y, theta_optimized, theta_history,
    #                      bias_plot_range, length_plot_range, 'cost_surface_with_path.png')