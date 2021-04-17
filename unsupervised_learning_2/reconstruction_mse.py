import matplotlib.pyplot as plt
import numpy as np

from gaussian_random_projection import GRPRunner
from ica import ICARunner
from pca import PCARunner
from util import get_africa_data, get_weather_data


def plot_mse(ys, num_components, title, seed=42):
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()
    xi = np.arange(1, num_components + 1, step=1)

    for i in range(len(ys)):
        y = ys[i]
        plt.plot(xi, y)
    plt.legend(["PCA", "ICA", "GRP"], loc="upper right")

    plt.xlabel("Number of Components")
    plt.xticks(
        np.arange(0, num_components + 1, step=1)
    )  # change from 0-based array index to 1-based human-readable label
    plt.ylabel("Reconstruction MSE")
    plt.title(f"{title} Reconstruction MSE for different numbers of components")

    ax.grid(axis="x")

    file_name = f"{title}-mse-seed_{seed}.png"
    plt.savefig("{}".format(file_name))
    plt.close()


def get_mse(x_train, title):
    pca_mse = []
    ica_mse = []
    for i in range(x_train.shape[1]):
        pca = PCARunner(
            x_train,
            None,
            n_components_percentage=0.9,
            title="PCA",
            seed=42,
            plot=False,
        )
        pca_mse.append(pca.get_mse(i + 1, x_train))
    for i in range(x_train.shape[1]):
        ica = ICARunner(
            x_train,
            None,
            title="ICA",
            seed=42,
            plot=False,
        )
        ica_mse.append(ica.get_mse(i + 1, x_train))
    grp = GRPRunner(
        x_train,
        None,
        None,
        None,
        n_iterations=20,
        title="GRP",
        seed=42,
        plot=False,
    )
    grp_mse = grp.get_num_components(return_mse=True)
    plot_mse([ica_mse, pca_mse, grp_mse], x_train.shape[1], title)


if __name__ == "__main__":
    AUSTRALIA = True
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.001, scaled=True
        )
        get_mse(aus_X_train, "Australian_Weather")

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.001, scaled=True
        )
        get_mse(aus_X_train, "Africa_Crisis")
