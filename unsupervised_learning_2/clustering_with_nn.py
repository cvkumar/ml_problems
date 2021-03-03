"""
5) Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms
(you've probably already done this), treating the clusters as if they were new features.
In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
-Run DR algorithms on dataset, then run clustering algorithms on result, then run neural network on that!
-1 (dataset) * 2 (Clustering algorithms) * 1 (Neural Network) => 2 NN

"""
import random

from util import get_weather_data, plot_clustering_labels_histogram
from sklearn.cluster import KMeans
import pandas as pd
from nn import NNRunner
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EM_NUM_CLUSTERS = 4
KM_NUM_CLUSTERS = 4


def run_nn_labels_em(seed=0):
    X, y = get_weather_data(split_train_test=False)
    em = GaussianMixture(
        covariance_type="diag", random_state=seed, n_components=EM_NUM_CLUSTERS
    )
    em.fit(X)
    labels = em.predict(X)
    plot_clustering_labels_histogram(
        title="NN EM Labels Histogram",
        labels=labels,
        output_path="results/clustering_with_nn/em/",
    )
    df_labels = pd.DataFrame()
    df_labels["labels"] = labels
    df_labels = pd.get_dummies(df_labels, columns=["labels"], drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df_labels, y, stratify=y, random_state=seed, test_size=0.5
    )
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 1),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [200],
        }
    ]
    nn = NNRunner(
        x_train,
        y_train,
        x_test=x_test,
        y_test=y_test,
        hyperparams=hyperparams,
        seed=seed,
        n_jobs=-1,
        plot=True,
        title="NN_EM_Labels",
        debug=True,
        output_path="results/clustering_with_nn/em/",
    )
    nn.run()


def run_nn_labels_kmeans(seed=0):
    X, y = get_weather_data(split_train_test=False)
    kmeans = KMeans(
        n_clusters=KM_NUM_CLUSTERS, init="k-means++", n_jobs=-1, random_state=seed
    )
    kmeans.fit(X)
    labels = kmeans.predict(X)
    plot_clustering_labels_histogram(
        title=f"NN KM Labels Histogram-seed{seed}",
        labels=labels,
        output_path="results/clustering_with_nn/km/",
    )
    df_labels = pd.DataFrame()
    df_labels["labels"] = labels
    df_labels = pd.get_dummies(df_labels, columns=["labels"], drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df_labels, y, stratify=y, random_state=seed, test_size=0.5
    )
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 1),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [200],
        }
    ]
    nn = NNRunner(
        x_train,
        y_train,
        x_test=x_test,
        y_test=y_test,
        hyperparams=hyperparams,
        seed=seed,
        n_jobs=-1,
        plot=True,
        title="NN_Kmeans_Labels",
        debug=True,
        output_path="results/clustering_with_nn/km/",
    )
    nn.run()


def run_nn_fake_data(seed=0):
    X, y = get_weather_data(split_train_test=False)
    df_labels = pd.DataFrame()
    df_labels["labels"] = [random.randint(0, 3) for i in range(10000)]
    df_labels = pd.get_dummies(df_labels, columns=["labels"], drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        df_labels, y, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 3),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [100],
        }
    ]
    nn = NNRunner(
        x_train,
        y_train,
        x_test=x_test,
        y_test=y_test,
        hyperparams=hyperparams,
        seed=seed,
        n_jobs=-1,
        plot=True,
        title="NN_Kmeans_Labels",
        debug=True,
        output_path="results/clustering_with_nn/fake/",
    )
    nn.run()


def run_nn_original(seed=0):
    aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(seed=seed)
    scaler = StandardScaler()
    scaler.fit(aus_X_train)
    aus_X_train = scaler.transform(aus_X_train)
    aus_X_test = scaler.transform(aus_X_test)
    aus_hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 1),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [10, 100],
        }
    ]
    aus_tester = NNRunner(
        aus_X_train,
        aus_y_train,
        aus_X_test,
        aus_y_test,
        aus_hyperparams,
        title="Australian_Weather_NN_Original",
        seed=seed,
        plot=True,
        debug=True,
        output_path="results/clustering_with_nn/original/",
    )
    aus_tester.run()


if __name__ == "__main__":
    run_nn_fake_data(seed=3)
    run_nn_labels_em(seed=3)

    run_nn_labels_kmeans(seed=3)
    run_nn_original(seed=3)
