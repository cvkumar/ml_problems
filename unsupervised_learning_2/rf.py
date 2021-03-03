import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from util import get_weather_data, get_africa_data

"""
RESOURCES: 
    https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f
"""


class RFRunner:
    def __init__(
        self,
        x,
        y,
        seed=42,
        n_estimators=100,
        n_jobs=7,
        plot=False,
        title="",
        debug=True,
    ):
        self.x = x
        self.y = y
        self.should_plot = plot
        self.seed = seed
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.title = title
        self.debug = debug

    def get_num_components(self):
        rfc = RandomForestClassifier(
            random_state=self.seed, n_estimators=self.n_estimators, n_jobs=self.n_jobs
        )
        selector = SelectFromModel(rfc).fit(self.x, self.y)
        selected_features_bool = selector.get_support()
        selected_features = [
            i
            for i in range(len(selected_features_bool))
            if selected_features_bool[i] == True
        ]
        n_components = len(selected_features)
        estimator = selector.estimator_
        feature_importance = estimator.feature_importances_
        if self.debug:
            print("n_components:", n_components)
            print("selected_features:", selected_features)
            print("estimator:", estimator)
            print("feature_importances:", feature_importance)

        if self.should_plot:
            self._plot(feature_importance, len(feature_importance), selector.threshold_)

        return selected_features

    def _plot(self, y, num_features, threshold):
        plt.rcParams["figure.figsize"] = (12, 6)
        fig, ax = plt.subplots()
        xi = np.arange(0, num_features, step=1)

        plt.plot(xi, y, marker="o", linestyle="--", color="b")

        plt.xlabel("Features")
        plt.xticks(
            np.arange(0, num_features, step=1)
        )  # change from 0-based array index to 1-based human-readable label
        plt.ylabel("Importance")
        plt.title(f"{self.title} Random Forest Feature Importance")

        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.text(0.5, 0.85, f"{threshold} cut-off threshold", color="red", fontsize=16)

        ax.grid(axis="x")
        file_name = f"{self.title}-seed_{self.seed}.png"
        plt.savefig("rf_results/{}".format(file_name))
        plt.close()


if __name__ == "__main__":
    AUSTRALIA = True
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.001, scaled=True
        )
        aus_tester = RFRunner(
            aus_X_train,
            aus_y_train,
            title="Australian_Weather_RF_Clustering",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester.get_num_components()

        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, scaled=True
        )
        aus_tester = RFRunner(
            aus_X_train,
            aus_y_train,
            title="Australian_Weather_RF_NN",
            seed=42,
            plot=True,
        )
        aus_n_components = aus_tester.get_num_components()

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.001, scaled=True
        )
        africa_tester = RFRunner(
            africa_X_train,
            africa_y_train,
            title="Africa_Crisis_RF_Clustering",
            seed=42,
            plot=True,
        )
        africa_n_components = africa_tester.get_num_components()
