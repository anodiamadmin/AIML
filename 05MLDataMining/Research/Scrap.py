import numpy as np
import pandas as pd


def read_and_one_hot_encode_abalone(file_path):
    df = pd.read_csv(file_path)
    df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True, dtype=float)
    return df_encoded


def cost_function(X, y, theta):
    """
    Computes the cost function for linear regression.
    Args:
        X: Feature matrix (m x n), where m is the number of samples and n is the number of features.
        y: Target vector (m x 1).
        theta: Parameter vector (n x 1).
    Returns:
        cost: The cost value.
    """
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    """
    Performs gradient descent to optimize the parameters of linear regression.
    Args:
        X: Feature matrix (m x n).
        y: Target vector (m x 1).
        theta: Initial parameter vector (n x 1).
        alpha: Learning rate.
        iterations: Number of iterations.
    Returns:
        theta: Optimized parameter vector (n x 1).
        cost_history: A list containing the cost after each iteration.
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        predictions = X @ theta
        error = predictions - y
        gradient = (1 / m) * (X.T @ error)
        theta = theta - alpha * gradient
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history


if __name__ == '__main__':

    # Main function to call read_and_one_hot_encode_abalone and display the head
    file_name = 'abalone.csv'
    abalone_data_encoded = read_and_one_hot_encode_abalone(file_name)

    if abalone_data_encoded is not None:
        print("Head of the one-hot encoded Abalone data with float values:")
        print(abalone_data_encoded.head())

    # Set hyperparameter
    alpha = 0.01
    iterations = 1000

    # Perform gradient descent
    theta_optimized, cost_history = gradient_descent(X, y, theta, alpha, iterations)

    # # Print results
    # print("Optimized parameters (theta): ")
    # print(theta_optimized)
    # print("Final cost:", cost_history[-1])
