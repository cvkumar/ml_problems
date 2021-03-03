"""
1) Run the clustering algorithms on the datasets and describe what you see. can choose your own measures of distance/similarity (will have to justify)
- K-means clustering
- Expectation Maximization
"""

from asgn3_unsupervised_learning.cluster_runners.expectation_maximization import (
    ExpectationMaximizationRunner,
)
from asgn3_unsupervised_learning.cluster_runners.kmeans import KMeansRunner
from util import get_weather_data, get_africa_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


def run_em_weather(seed=0):
    X_train, X_test, y_train, y_test = get_weather_data(10000, test_size=0.001)
    tester = ExpectationMaximizationRunner(
        X_train,
        y_train,
        title="Australian_Weather_EM",
        seed=seed,
        clusters=list(range(2, 20)),
        plot=True,
        chosen_cluster=5,
        write_labels=False,
        compute_silhouette=True,
    )
    tester.run()


def run_em_africa(seed=0, chosen_cluster=5):
    X_train, X_test, y_train, y_test = get_africa_data(test_size=0.001)
    tester = ExpectationMaximizationRunner(
        X_train,
        y_train,
        title="Africa_Crisis_EM",
        seed=seed,
        clusters=list(range(2, 20)),
        plot=True,
        chosen_cluster=chosen_cluster,
        write_labels=False,
        compute_silhouette=True,
    )
    tester.run()


def run_km_weather(seed=0):
    X_train, X_test, y_train, y_test = get_weather_data(
        10000, test_size=0.001, one_hot_encode=True
    )
    kmeans = KMeansRunner(
        X_train,
        y_train,
        list(range(2, 20)),
        should_plot=True,
        write_labels=True,
        title="Australian_Weather_KMeans",
        chosen_cluster=5,
        n_init=10,
        max_iters=500,
        seed=seed,
        compute_silhouette=True,
    )
    kmeans.run()


def run_km_africa():
    X_train, X_test, y_train, y_test = get_weather_data(
        10000, test_size=0.001, one_hot_encode=True
    )
    kmeans = KMeansRunner(
        X_train,
        y_train,
        list(range(2, 20)),
        should_plot=True,
        write_labels=True,
        title="Africa_Crisis_KMeans",
        chosen_cluster=5,
        n_init=10,
        max_iters=500,
        seed=0,
        compute_silhouette=False,
    )
    kmeans.run()


if __name__ == "__main__":
    run_em_weather(seed=1)
    run_km_weather(seed=1)

    # run_em_africa()
    # run_km_africa()
