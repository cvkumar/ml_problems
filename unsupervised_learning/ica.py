import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from util import get_weather_data, get_africa_data

"""
RESOURCES: 
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html?highlight=ica#sklearn.decomposition.FastICA
"""


class ICARunner:
    def __init__(self, x, y, seed=42, plot=False, title="", debug=True):
        self.x = x
        self.y = y
        self.should_plot = plot
        self.seed = seed
        self.title = title
        self.debug = debug

    def get_num_components(self):
        n_components_list = list(
            range(1, self.x.shape[1] + 1)
        )  # May want do do number of features-1 so that we don't use all the features
        mean_kurtosis = []
        for n in n_components_list:
            ica = FastICA(random_state=self.seed, n_components=n)
            x_transform = pd.DataFrame(ica.fit_transform(self.x))
            kurtosis = x_transform.kurt(axis=0)
            mean_kurtosis.append(kurtosis.abs().mean())
        num_components = mean_kurtosis.index(max(mean_kurtosis[:-1])) + 1

        if self.debug:
            print(f"mean_kurtosis: {mean_kurtosis}")
            print(f"num components: {num_components}")

        if self.should_plot:
            self._plot(mean_kurtosis, len(mean_kurtosis))

        return num_components

    def _plot(self, y, num_components):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, num_components + 1, step=1)

        plt.plot(xi, y, marker="o", linestyle="--", color="b")

        plt.xlabel("Number of Components")
        plt.xticks(
            np.arange(0, num_components + 1, step=1)
        )  # change from 0-based array index to 1-based human-readable label
        plt.ylabel("Mean Kurtosis")
        plt.title(f"{self.title} Mean kurtosis for different numbers of components")

        ax.grid(axis="x")
        file_name = f"{self.title}-seed_{self.seed}.png"
        plt.savefig("ica_results/{}".format(file_name))
        plt.close()


if __name__ == "__main__":
    AUSTRALIA = True
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.001, scaled=True
        )
        aus_tester = ICARunner(
            aus_X_train,
            aus_y_train,
            title="Australian_Weather_ICA_Clustering",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester.get_num_components()
        aus_ica = FastICA(n_components=aus_n_components)
        aus_X_transform = aus_ica.fit_transform(aus_X_train)
        aus_inverse = aus_ica.inverse_transform(aus_X_transform)
        aus_mse = np.sum(np.square(aus_X_train - aus_inverse)) / aus_inverse.size
        print("MSE for dataset 1: ", aus_mse)

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.001, scaled=True
        )
        africa_tester = ICARunner(
            africa_X_train,
            africa_y_train,
            title="Africa_Crisis_ICA_Clustering",
            seed=42,
            plot=True,
        )
        africa_n_components = africa_tester.get_num_components()
        africa_ica = FastICA(n_components=africa_n_components)
        africa_X_transform = africa_ica.fit_transform(africa_X_train)
        africa_inverse = africa_ica.inverse_transform(africa_X_transform)
        africa_mse = (
            np.sum(np.square(africa_X_train - africa_inverse)) / africa_inverse.size
        )
        print("MSE for dataset 2: ", africa_mse)
