"""
k-Nearest Neighbors. You should "implement" (the quotes mean I don't mean it: steal the code) kNN. Use different values of k.
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from util import (
    plot_validation_curve,
    get_weather_data,
    plot_learning_curve,
    get_africa_data,
)


def knn_weather():
    X_train, X_test, y_train, y_test = get_weather_data(10000)
    plot_validation_curve(
        KNeighborsClassifier(),
        "Dataset 1",
        X_train,
        y_train,
        param_name="n_neighbors",
        param_range=[
            x for x in range(10, 60) if x % 2 == 1
        ],  # Neighbors for only odd numbers
        n_jobs=-1,
    )
    grid_params = {
        "n_neighbors": [x for x in range(10, 25) if x % 2 == 1],
        "weights": [
            "uniform"
        ],  # adding distance as metric overfits score to 1.0 on the training data
        "metric": ["manhattan"],
    }
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs_results = gs.fit(X_train, y_train)
    test_score = gs_results.best_estimator_.score(X_test, y_test)
    print(
        f"best_score: {gs_results.best_score_}\n, best_estimator: {gs_results.best_estimator_}\n, best_params: {gs_results.best_params_}"
        f"best score on test set: {test_score}"
    )
    plot_learning_curve(
        gs_results.best_estimator_, "KNN Classifier Dataset 1", X_train, y_train
    )
    # plt.show()


def knn_africa():
    X_train, X_test, y_train, y_test = get_africa_data()
    plot_validation_curve(
        KNeighborsClassifier(),
        "Dataset 2",
        X_train,
        y_train,
        param_name="n_neighbors",
        param_range=[
            x for x in range(3, 30) if x % 2 == 1
        ],  # Neighbors for only odd numbers
        n_jobs=-1,
    )
    grid_params = {
        "n_neighbors": [x for x in range(3, 25) if x % 2 == 1],
        "weights": [
            "uniform"
        ],  # adding distance as metric overfits score to 1.0 on the training data
        "metric": ["manhattan", "euclidean"],
    }
    gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=3, n_jobs=-1)
    gs_results = gs.fit(X_train, y_train)
    test_score = gs_results.best_estimator_.score(X_test, y_test)
    print(
        f"best_score: {gs_results.best_score_}\n, best_estimator: {gs_results.best_estimator_}\n, best_params: {gs_results.best_params_}"
        f"best score on test set: {test_score}"
    )
    plot_learning_curve(
        gs_results.best_estimator_, "KNN Classifier Dataset 2", X_train, y_train
    )
    # plt.show()


if __name__ == "__main__":
    knn_weather()

    knn_africa()
