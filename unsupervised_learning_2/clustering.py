"""
1) Run the clustering algorithms on the datasets and describe what you see. can choose your own measures of distance/similarity (will have to justify)
- K-means clustering
- Expectation Maximization
"""

from expectation_maximization import ExpectationMaximizationRunner
from kmeans import KMeansRunner
from util import get_weather_data, get_africa_data


# Num Clusters based on of Silhouette Score is 4
def run_em_weather(seed=0):
    x, y = get_weather_data(10000, split_train_test=False)
    tester = ExpectationMaximizationRunner(
        x,
        y,
        title="Australian_Weather_EM",
        seed=seed,
        clusters=list(range(2, 20)),
        plot=True,
        chosen_cluster=4,
        write_labels=False,
        compute_silhouette=True,
        result_path="results/clustering/weather/",
    )
    tester.run()


# BASED ON SILHOUETTE SCORE BEST RESULT IS 3
def run_em_africa(seed=0, chosen_cluster=3):
    x, y = get_africa_data(split_train_test=False)
    tester = ExpectationMaximizationRunner(
        x,
        y,
        title="Africa_Crisis_EM",
        seed=seed,
        clusters=list(range(2, 20)),
        plot=True,
        chosen_cluster=chosen_cluster,
        write_labels=False,
        compute_silhouette=True,
        result_path="results/clustering/africa/",
    )
    tester.run()


# BASED ON ELBOW AND SILHOUETTE CLUSTER SIZE 4
def run_km_weather(seed=0):
    x, y = get_weather_data(10000, split_train_test=False)
    kmeans = KMeansRunner(
        x,
        y,
        list(range(2, 20)),
        should_plot=True,
        write_labels=False,
        title="Australian_Weather_KMeans",
        chosen_cluster=4,
        n_init=10,
        max_iters=500,
        seed=seed,
        compute_silhouette=True,
        result_path="results/clustering/weather/",
    )
    kmeans.run()


# BASED ON ELBOW AND SILHOUETTE CLUSTER SIZE 3
def run_km_africa(seed=0):
    x, y = get_africa_data(split_train_test=False)
    kmeans = KMeansRunner(
        x,
        y,
        list(range(2, 20)),
        should_plot=True,
        write_labels=False,
        title="Africa_Crisis_KMeans",
        chosen_cluster=5,
        n_init=10,
        max_iters=500,
        seed=seed,
        compute_silhouette=True,
        result_path="results/clustering/africa/",
    )
    kmeans.run()


if __name__ == "__main__":
    run_em_weather(seed=1)
    run_km_weather(seed=1)

    run_em_africa(seed=1)
    run_km_africa(seed=1)
