"""
This should be done in a way such that you can swap out kernel functions. I'd like to see at least two

HYPERPARAMETERS:
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

Kernel: The kernel functions return the inner product between two points in a suitable feature space.
Thus by defining a notion of similarity, with little computational cost even in very high-dimensional spaces.

C: parameter trades off correct classification of training examples against maximization of the decision function’s 
margin. For larger values of C, a smaller margin will be accepted if the decision function is better at 
classifying all training points correctly. A lower C will encourage a larger margin, therefore a simpler decision 
function, at the cost of training accuracy. In other words``C`` behaves as a regularization parameter in the SVM.

"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn import svm

from util import (
    plot_validation_curve,
    plot_learning_curve,
    get_africa_data,
    get_weather_data,
)


def _get_range_of_powers_of_ten(start_range, end_range):
    return [1 * 10 ** x for x in range(start_range, end_range)]


def svm_weather():
    X_train, X_test, y_train, y_test = get_weather_data(10000)
    plot_validation_curve(
        svm.SVC(),
        "Dataset 1",
        X_train,
        y_train,
        param_name="C",
        param_range=_get_range_of_powers_of_ten(2, 6),
        n_jobs=-1,
        semilog=True,
    )
    rbf = {
        "C": _get_range_of_powers_of_ten(4, 6),
        "kernel": ["rbf"],
    }
    poly = {
        "C": _get_range_of_powers_of_ten(4, 6),
        "kernel": ["poly"],
    }
    sigmoid = {
        "C": _get_range_of_powers_of_ten(4, 6),
        "kernel": ["sigmoid"],
    }
    grid_params = [poly, sigmoid, rbf]
    gs = GridSearchCV(svm.SVC(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs_results = gs.fit(X_train, y_train)
    test_score = gs_results.best_estimator_.score(X_test, y_test)
    print(
        f"best_score: {gs_results.best_score_}\n, best_estimator: {gs_results.best_estimator_}\n, best_params: {gs_results.best_params_}\n"
        f"best score on test set: {test_score}\n"
    )
    plot_learning_curve(
        gs_results.best_estimator_, "SVM Classifier Dataset 1", X_train, y_train
    )
    # plt.show()


def svm_africa():
    X_train, X_test, y_train, y_test = get_africa_data()
    plot_validation_curve(
        svm.SVC(),
        "Dataset 2",
        X_train,
        y_train,
        param_name="C",
        param_range=_get_range_of_powers_of_ten(-5, 14),
        n_jobs=-1,
        semilog=True,
    )
    # plot shows that best value for C is between 10^0 and 10^6
    # NOTE: Plot done without changing the value of gamma or kernel function
    rbf = {
        "C": _get_range_of_powers_of_ten(0, 6),
        "kernel": ["rbf"],
    }
    poly = {
        "C": _get_range_of_powers_of_ten(0, 6),
        "kernel": ["poly"],
    }
    sigmoid = {
        "C": _get_range_of_powers_of_ten(0, 6),
        "kernel": ["sigmoid"],
    }
    grid_params = [poly, sigmoid, rbf]
    gs = GridSearchCV(svm.SVC(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs_results = gs.fit(X_train, y_train)
    test_score = gs_results.best_estimator_.score(X_test, y_test)
    print(
        f"best_score: {gs_results.best_score_}\n, best_estimator: {gs_results.best_estimator_}\n, best_params: {gs_results.best_params_}\n"
        f"best score on test set: {test_score}\n"
    )
    plot_learning_curve(
        gs_results.best_estimator_, "SVM Classifier Dataset 2", X_train, y_train
    )
    # plt.show()


if __name__ == "__main__":
    svm_africa()

    svm_weather()
