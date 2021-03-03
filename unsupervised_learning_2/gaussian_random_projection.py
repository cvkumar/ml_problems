import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC

from util import get_weather_data, get_africa_data

"""
RESOURCES: 
    https://scikit-learn.org/stable/modules/random_projection.html
    https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
    http://blog.yhat.com/posts/sparse-random-projections.html
"""


class GRPRunner:
    def __init__(
        self,
        x,
        y,
        x_test,
        y_test,
        n_iterations,
        C=1.0,
        seed=42,
        plot=False,
        title="",
        debug=True,
    ):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.C = C
        self.should_plot = plot
        self.num_iterations = n_iterations
        self.seed = seed
        self.title = title
        self.debug = debug

    def get_num_components(self):
        n_components_list = list(
            range(1, self.x.shape[1])
        )  # May want do do features -1 so that we don't use all the features
        mean_reconstruction_error = []
        for n in n_components_list:
            reconstruction_error = []
            for i in range(self.num_iterations):
                grp = GaussianRandomProjection(random_state=i, n_components=n)
                x_transform = grp.fit_transform(self.x)
                pseudo_inverse = np.linalg.pinv(grp.components_.T)
                x_reconstructed = np.dot(x_transform, pseudo_inverse)
                reconstruction_error.append(np.square(self.x - x_reconstructed))
            mean_reconstruction_error.append(np.nanmean(reconstruction_error))
        num_components = (
            mean_reconstruction_error.index(min(mean_reconstruction_error)) + 1
        )

        if self.debug:
            print(f"mean squared reconstruction error: {mean_reconstruction_error}")
            print(f"num components: {num_components}")

        if self.should_plot:
            self._plot(mean_reconstruction_error, len(mean_reconstruction_error))

        return num_components

    def get_num_components_svm(self):
        """
        http://blog.yhat.com/posts/sparse-random-projections.html
        :return:
        """
        model = LinearSVC(random_state=self.seed)
        model.fit(self.x, self.y)
        baseline = metrics.accuracy_score(model.predict(self.x_test), self.y_test)

        n_components_list = list(
            range(1, self.x.shape[1])
        )  # May want do do features -1 so that we don't use all the features
        mean_accuracies = []
        # loop over the projection sizes
        for n in n_components_list:
            accuracies = []
            for i in range(self.num_iterations):
                if self.debug:
                    print(f"n:{n} i:{i}")
                # create the random projection
                grp = GaussianRandomProjection(random_state=i, n_components=n)
                x_transform = grp.fit_transform(self.x)

                # train a classifier on the sparse random projection
                model = LinearSVC(random_state=self.seed)
                model.fit(x_transform, self.y)

                # evaluate the model and update the list of accuracies
                test = grp.transform(self.x_test)
                accuracies.append(
                    metrics.accuracy_score(model.predict(test), self.y_test)
                )
            mean_accuracies.append(np.nanmean(accuracies))

        if self.should_plot:
            # create the figure
            plt.rcParams["figure.figsize"] = (12, 6)
            fig, ax = plt.subplots()
            xi = np.arange(1, len(n_components_list) + 1, step=1)

            # plot the baseline and random projection accuracies
            plt.plot(n_components_list, [baseline] * len(mean_accuracies), color="r")
            plt.plot(xi, mean_accuracies, marker="o", linestyle="--", color="b")

            plt.xlabel("Number of Components")
            plt.xticks(
                np.arange(0, len(n_components_list) + 1, step=1)
            )  # change from 0-based array index to 1-based human-readable label
            plt.ylabel("Accuracy")
            plt.title(f"{self.title} Accuracy of Projection")

            ax.grid(axis="x")

            file_name = f"{self.title}-svm-seed_{self.seed}.png"
            plt.savefig("grp_results/{}".format(file_name))
            plt.close()

    def _plot(self, y, num_components):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(1, num_components + 1, step=1)

        plt.plot(xi, y, marker="o", linestyle="--", color="b")
        plt.plot(xi, [0.5] * len(xi), color="r")
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
        plt.savefig("grp_results/{}".format(file_name))
        plt.close()


if __name__ == "__main__":
    AUSTRALIA = True
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.001, scaled=True
        )
        aus_tester_clustering = GRPRunner(
            aus_X_train,
            aus_y_train,
            aus_X_test,
            aus_y_test,
            n_iterations=20,
            title="Australian_Weather_GRP_Clustering",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester_clustering.get_num_components()
        aus_tester_clustering.get_num_components_svm()

        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, scaled=True
        )
        aus_tester_nn = GRPRunner(
            aus_X_train,
            aus_y_train,
            aus_X_test,
            aus_y_test,
            n_iterations=20,
            title="Australian_Weather_GRP_NN",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester_nn.get_num_components()
        aus_tester_nn.get_num_components_svm()

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.001, scaled=True
        )
        africa_tester = GRPRunner(
            africa_X_train,
            africa_y_train,
            africa_X_test,
            africa_y_test,
            C=0.1,  # this doesn't even seem to matter since it's always 1.0 accuracy :\
            n_iterations=20,
            title="Africa_Crisis_GRP_Clustering",
            seed=42,
            plot=True,
        )
        africa_n_components = africa_tester.get_num_components()
        africa_tester.get_num_components_svm()
