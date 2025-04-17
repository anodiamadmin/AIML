import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from sklearn import datasets, metrics
# roc_auc_score: Area Under Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
# Evaluation metric for binary classification (and can be extended to multiclass).
from sklearn.metrics import (mean_squared_error, accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def read_data(run_num):
    # Source: Pima-Indian diabetes dataset
    data_in = genfromtxt("data/pima.csv", delimiter=",")
    data_inputx = data_in[:, 0:8]  # All features: columns 0 to 7
    data_inputy = data_in[:, -1]   # Target/label: last column
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy,
                                                        test_size=0.4, random_state=run_num)
    return x_train, x_test, y_train, y_test


# Create FNN: Use multilayer perceptron (MLP) classifier from SciKit learn.
# The parameter 'type_model' selects one of the 3 networks defined from sklearn.neural_network
def scipy_nn(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num):
    if type_model == 0:  # SGD (Stochastic Gradient Descent)
        nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100, solver='sgd', learning_rate_init=learn_rate)
    elif type_model == 1:  # Adam
        nn = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100, solver='adam', learning_rate_init=learn_rate)
    elif type_model == 2:  # SGD with 2 hidden layers
        nn = MLPClassifier(hidden_layer_sizes=(hidden, hidden), random_state=run_num, max_iter=100, solver='sgd', learning_rate_init=learn_rate)
    else:
        print('no model')
        return
    nn.fit(x_train, y_train)  # Train the model
    y_pred_test = nn.predict(x_test)  # Make predictions
    y_pred_train = nn.predict(x_train)
    acc_test = accuracy_score(y_pred_test, y_test)
    acc_train = accuracy_score(y_pred_train, y_train)
    cm = confusion_matrix(y_pred_test, y_test)
    return acc_test


def main():
    x_train, x_test, y_train, y_test = read_data(run_num=1)
    # Display the first 5 rows
    print(f"Training features (x_train):\n{x_train[:5]}\nTraining labels (y_train):\n{y_train[:5]}")
    print(f"\nTest features (x_test):\n{x_test[:5]}\nTest labels (y_test):{y_test[:5]}")
    # Execute the tasks: Select 0, 1 0r 2 fot type_model and run with these options
    type_model = 0
    max_expruns = 10
    acc_arr = np.zeros(max_expruns)
    learn_rate = 0.01
    hidden = 8
    if type_model == 0:
        print ( '1 hidden layer with SCD')
    elif type_model == 1:
        print('1 hidden layer with ADAM')
    elif type_model == 2:
        print ('2 hidden layer with SCD')
    learn_ratevec = np.arange(0.01, 0.1, 0.02)
    print('No of learning rates: ', learn_ratevec.shape[0])
    print('Vector of learning rates: ', learn_ratevec)
    for learn_rate in learn_ratevec:
        print('\n\n------- Running for learning rate:', learn_rate, '-------------')
        for run_num in range(0, max_expruns):
            x_train, x_test, y_train, y_test = read_data(run_num)
            acc = scipy_nn(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num)
            print(f"Accuracy in run_num: {run_num+1} is = {acc}")
            acc_arr[run_num] = acc
        print(f"{hidden} hidden layers.\nMean accuracy = {np.mean(acc_arr)}, Std of accuracy = {np.std(acc_arr)}")


if __name__ == "__main__":
    main()