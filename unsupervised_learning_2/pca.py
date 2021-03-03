import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from util import get_weather_data, get_africa_data

"""
RESOURCES: 
    https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/
    https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    https://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
"""


class PCARunner:
    def __init__(
        self, x, y, n_components_percentage=0.95, seed=42, plot=False, title=""
    ):
        self.x = x
        self.y = y
        self.percentage = n_components_percentage
        self.should_plot = plot
        self.seed = seed
        self.title = title

    def get_num_components(self):
        pca = PCA(random_state=self.seed, n_components=self.percentage)
        pca.fit(self.x)

        y_cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = y_cumsum_variance.shape[0]

        mse = []
        for n in range(self.x.shape[1]):
            mse.append(self.get_mse(n, self.x))
        if self.should_plot:
            self._plot(y_cumsum_variance, num_components)
            self._plot_mse(mse, self.x.shape[1])
            self._plot_distribution(pca)
        return num_components

    def get_mse(self, n_components, x_train):
        pca = PCA(n_components=n_components)
        x_transform = pca.fit_transform(x_train)
        x_inverse = pca.inverse_transform(x_transform)
        mse = np.mean(np.square(x_train - x_inverse))
        return mse

    def _plot(self, y, num_components):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, num_components + 1, step=1)

        plt.ylim(0.0, 1.1)
        plt.plot(xi, y, marker="o", linestyle="--", color="b")

        plt.xlabel("Number of Components")
        plt.xticks(
            np.arange(0, num_components + 1, step=1)
        )  # change from 0-based array index to 1-based human-readable label
        plt.ylabel("Cumulative variance (%)")
        plt.title(f"{self.title} number of components needed to explain variance")

        plt.axhline(y=self.percentage, color="r", linestyle="-")
        plt.text(
            0.5,
            0.85,
            f"{self.percentage * 100}% cut-off threshold",
            color="red",
            fontsize=16,
        )

        ax.grid(axis="x")
        file_name = f"{self.title}-seed_{self.seed}.png"
        plt.savefig("pca_results/{}".format(file_name))
        plt.close()

    def _plot_mse(self, y, num_components):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, num_components + 1, step=1)

        plt.plot(xi, y, marker="o", linestyle="--", color="b")

        plt.xlabel("Number of Components")
        plt.xticks(
            np.arange(0, num_components + 1, step=1)
        )  # change from 0-based array index to 1-based human-readable label
        plt.ylabel("Reconstruction MSE")
        plt.title(
            f"{self.title} Reconstruction MSE for different numbers of components"
        )

        ax.grid(axis="x")

        file_name = f"{self.title}-mse-seed_{self.seed}.png"
        plt.savefig("pca_results/{}".format(file_name))
        plt.close()

    def _plot_distribution(self, pca):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, pca.n_features_ + 1, step=1)

        legend = []
        for i in range(pca.n_components_):
            legend.append(str(i + 1))
            plt.bar(xi, pca.components_[i])

        plt.legend(legend, loc="upper left")

        plt.xlabel("Features")
        plt.xticks(
            np.arange(0, pca.n_features_ + 1, step=1)
        )  # change from 0-based array index to 1-based human-readable label
        plt.ylabel("Feature Weight")
        plt.title(f"{self.title} Distribution of Eigenvalues")

        ax.grid(axis="x")

        file_name = f"{self.title}-distribution-seed_{self.seed}.png"
        plt.savefig("pca_results/{}".format(file_name))
        plt.close()


def get_reduced_dataset(x, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)


if __name__ == "__main__":
    AUSTRALIA = True
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.001, scaled=True
        )
        aus_tester = PCARunner(
            aus_X_train,
            aus_y_train,
            n_components_percentage=0.9,
            title="Australian_Weather_PCA_Clustering",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester.get_num_components()
        aus_pca = PCA(n_components=aus_n_components)
        aus_X_transform = aus_pca.fit_transform(
            aus_X_train
        )  # this is what we need to pass to clustering and nn
        aus_inverse = aus_pca.inverse_transform(aus_X_transform)
        aus_mse = np.mean(np.square(aus_X_train - aus_inverse))
        print("MSE for dataset 1 Clustering: ", aus_mse)

        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, scaled=True
        )
        aus_tester = PCARunner(
            aus_X_train,
            aus_y_train,
            n_components_percentage=0.9,
            title="Australian_Weather_PCA_NN",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester.get_num_components()
        aus_pca = PCA(n_components=aus_n_components)
        aus_X_transform = aus_pca.fit_transform(
            aus_X_train
        )  # this is what we need to pass to clustering and nn
        aus_inverse = aus_pca.inverse_transform(aus_X_transform)
        aus_mse = np.mean(np.square(aus_X_train - aus_inverse))
        print("MSE for dataset 1 NN: ", aus_mse)

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.001, scaled=True
        )
        africa_tester = PCARunner(
            africa_X_train,
            africa_y_train,
            n_components_percentage=0.9,
            title="Africa_Crisis_PCA_Clustering",
            seed=42,
            plot=True,
        )
        africa_n_components = africa_tester.get_num_components()
        africa_pca = PCA(n_components=africa_n_components)
        africa_X_transform = africa_pca.fit_transform(
            africa_X_train
        )  # this is what we need to pass to clustering and nn
        africa_inverse = africa_pca.inverse_transform(africa_X_transform)
        africa_mse = np.mean(np.square(africa_X_train - africa_inverse))
        print("MSE for dataset 2: ", africa_mse)
