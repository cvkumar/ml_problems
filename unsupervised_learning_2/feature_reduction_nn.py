"""
4) Apply the dimensionality reduction algorithms to one of your datasets from assignment #1
(if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this)
and rerun your neural network learner on the newly projected data.
-One dataset, run 4 DR algorithms on it, run neural network on result
-1 (dataset) * 4 (DRs) * 1 (Neural Network) => 4 NN

"""

from nn import NNRunner
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection

from util import get_weather_data

AUS_X_TRAIN, AUS_X_TEST, AUS_Y_TRAIN, AUS_Y_TEST = get_weather_data(
    10000, test_size=0.5, scaled=True
)
HYPERPARAMS = [
    {
        "hidden_layer_sizes": range(10, 30),
        # 'activation': ['tanh', 'relu'],
        # 'solver': ['sgd', 'adam'],
        # 'alpha': [0.0001, 0.05],
        # 'learning_rate': ['constant', 'adaptive'],
        "max_iter": [10, 100],
    }
]


def run_ica_nn(seed=42, n_components=8):
    """
    Get feature reduced data using ICA
    Use results from ica.py to get num_components
    """
    ica = FastICA(n_components=n_components)
    x_transform = ica.fit_transform(AUS_X_TRAIN)
    x_test_transform = ica.transform(AUS_X_TEST)
    nn = NNRunner(
        x_transform,
        AUS_Y_TRAIN,
        x_test_transform,
        AUS_Y_TEST,
        hyperparams=HYPERPARAMS,
        title="Australian_Weather_ICA",
        seed=seed,
        plot=True,
        debug=True,
    )
    nn.run()


def run_pca_nn(seed=42, n_components=12):
    """
    Get feature reduced data using PCA
    Use results from pca.py to get num_components
    """
    pca = PCA(n_components=n_components)
    x_transform = pca.fit_transform(AUS_X_TRAIN)
    x_test_transform = pca.transform(AUS_X_TEST)
    nn = NNRunner(
        x_transform,
        AUS_Y_TRAIN,
        x_test_transform,
        AUS_Y_TEST,
        hyperparams=HYPERPARAMS,
        title="Australian_Weather_PCA",
        seed=seed,
        plot=True,
        debug=True,
    )
    nn.run()


def run_grp_weather(seed=42, n_components=15):
    """
    Get feature reduced data using PCA
    Use results from pca.py to get num_components
    """
    grp = GaussianRandomProjection(n_components=n_components)
    x_transform = grp.fit_transform(AUS_X_TRAIN)
    x_test_transform = grp.transform(AUS_X_TEST)
    nn = NNRunner(
        x_transform,
        AUS_Y_TRAIN,
        x_test_transform,
        AUS_Y_TEST,
        hyperparams=HYPERPARAMS,
        title=f"Australian_Weather_GRP-{n_components}-components",
        seed=seed,
        plot=True,
        debug=True,
    )
    nn.run()


def run_rf_weather(seed=42, components=[3, 7, 12, 13, 14, 15]):
    """
    Get feature reduced data using PCA
    Use results from pca.py to get num_components
    """
    x_transform = AUS_X_TRAIN[:, components]
    x_test_transform = AUS_X_TEST[:, components]
    nn = NNRunner(
        x_transform,
        AUS_Y_TRAIN,
        x_test_transform,
        AUS_Y_TEST,
        hyperparams=HYPERPARAMS,
        title="Australian_Weather_RF",
        seed=seed,
        plot=True,
        debug=True,
    )
    nn.run()


if __name__ == "__main__":
    run_ica_nn()
    run_pca_nn()
    run_grp_weather(n_components=15)  # test with both n=15 and n_components=19
    run_grp_weather(n_components=19)  # test with both n=15 and n_components=19
    run_rf_weather()
