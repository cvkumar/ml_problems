"""
5) Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms
(you've probably already done this), treating the clusters as if they were new features.
In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
-Run DR algorithms on dataset, then run clustering algorithms on result, then run neural network on that!
-1 (dataset) * 2 (Clustering algorithms) * 1 (Neural Network) => 2 NN

"""

from asgn3_unsupervised_learning.nn import NNRunner
from util import get_weather_data
from sklearn.cluster import KMeans
import pandas as pd


def run_nn_labels_em(seed=0):
    X, y = get_weather_data(10000, test_size=0.001, split_train_test=False)
    from sklearn.mixture import GaussianMixture

    em = GaussianMixture(covariance_type="diag", random_state=seed, n_components=4)
    em.fit(X)
    labels = em.predict(X)
    df_labels = pd.DataFrame()
    df_labels["labels"] = labels
    df_labels = pd.get_dummies(df_labels, columns=["labels"], drop_first=True)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        df_labels, y, stratify=y, test_size=0.5, random_state=43
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 2),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [10, 100],
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
    )
    nn.neural_network()


def run_nn_labels_kmeans(seed=0):
    X, y = get_weather_data(10000, test_size=0.001, split_train_test=False)
    kmeans = KMeans(n_clusters=4, init="k-means++", n_jobs=-1, random_state=3)
    labels = kmeans.fit_predict(X)
    df_labels = pd.DataFrame()
    df_labels["labels"] = labels
    df_labels = pd.get_dummies(df_labels, columns=["labels"], drop_first=True)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        df_labels, y, stratify=y, test_size=0.5, random_state=43
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 5),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [10, 100],
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
    )
    nn.neural_network()


def run_nn_original_weather():
    # run_nn_labels_weather()
    aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
        10000, test_size=0.5
    )
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(aus_X_train)
    aus_X_train = scaler.transform(aus_X_train)
    aus_X_test = scaler.transform(aus_X_test)
    aus_hyperparams = [
        {
            "hidden_layer_sizes": range(10, 30, 5),
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
        seed=42,
        plot=True,
        debug=True,
    )
    aus_tester.neural_network()


if __name__ == "__main__":
    run_nn_labels_em(seed=10)

    # x_train, x_test, y_train, y_test = train_test_split(df_with_all, y, stratify=y, test_size=.5, random_state=42)
    #
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    #
    # hyperparams = [{
    #     'hidden_layer_sizes': range(10, 30),
    #     # 'activation': ['tanh', 'relu'],
    #     # 'solver': ['sgd', 'adam'],
    #     # 'alpha': [0.0001, 0.05],
    #     # 'learning_rate': ['constant', 'adaptive'],
    #     'max_iter': [10, 100]
    # }]
    #
    # nn1 = NNRunner(
    #     x_train,
    #     y_train, x_test=x_test, y_test=y_test, hyperparams=hyperparams, seed=0, n_jobs=-1, plot=True,
    #     title="NN with labels and original data"
    # )
    # nn1.neural_network()
