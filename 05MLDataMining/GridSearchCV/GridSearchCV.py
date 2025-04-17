import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV  # ******* Grid Search Cross Validation
import random
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

data_in = genfromtxt("./data/pima.csv", delimiter=",")

data_inputx = data_in[:, 0:8]  # all features 0, 1, 2, 3, 4, 5, 6, 7
data_inputy = data_in[:, -1]  # this is target - so that last col is selected from data

# Data Exploration, Transformations, etc. here...

# split to training and test sets
x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=23)

# model generation object
mlp = MLPClassifier(max_iter=100)

# Parameter Space
parameter_space = {
    'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (50,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive'],
}

# Search through parameter space, build models for different parameters
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)  # n_jobs=#of machine cores, cv=Cross Val Iterations
clf.fit(x_train, y_train)   # clf=classifier, fit=train, on train data only

# Best paramete set
print('Best parameters discovered:\n', clf.best_params_)

# y_true, y_pred = y_test, clf.predict(x_test)
#
# print('\nReport results on the test set:')
# print(classification_report(y_true, y_pred))
#
# acc_test = accuracy_score(y_pred, y_true)
#
# y_pred_train = clf.predict(x_train)
# acc_train = accuracy_score(y_pred_train, y_train)
#
# cm = confusion_matrix(y_pred, y_true)
# # print(cm, 'is confusion matrix')
#
# print("acc_test: ", acc_test)
# # All results
# print("------  All Results ------- ")
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("      %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
# print("------  End All Results ------- ")
