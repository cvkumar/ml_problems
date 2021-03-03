from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from util import (
    get_weather_data,
    plot_validation_curve,
    plot_learning_curve,
    get_africa_data,
)


class NNRunner:
    def __init__(
        self,
        x,
        y,
        x_test,
        y_test,
        hyperparams,
        seed=42,
        n_jobs=7,
        plot=False,
        title="",
        debug=True,
        output_path=None,
    ):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.should_plot = plot
        self.seed = seed
        self.hyperparams = hyperparams
        self.n_jobs = n_jobs
        self.title = title
        self.debug = debug
        self.output_path = output_path

    def run(self):
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        if self.should_plot:
            # generate validation scores over a range of param values
            plot_validation_curve(
                MLPClassifier(),
                f"{self.title} Neural Network Classifier",
                self.x,
                self.y,
                "hidden_layer_sizes",
                range(1, 100),
                filename=self.title,
                n_jobs=self.n_jobs,
                seed=self.seed,
                output_path=self.output_path,
            )

        scores = ["precision", "recall"]

        for score in scores:
            if self.debug:
                print("# Tuning hyper-parameters for %s" % score)
                print()

            clf = GridSearchCV(
                MLPClassifier(),
                self.hyperparams,
                scoring="%s_macro" % score,
                n_jobs=self.n_jobs,
            )
            clf_results = clf.fit(self.x, self.y)
            if self.debug:
                print("Best parameters set found on development set:")
                print()
                print(clf_results.best_params_)
                print()
                print("Grid scores on development set:")
                print()
            means = clf_results.cv_results_["mean_test_score"]
            stds = clf_results.cv_results_["std_test_score"]
            if self.debug:
                for mean, std, params in zip(
                    means, stds, clf_results.cv_results_["params"]
                ):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
                print()

                print("Detailed classification report:")
                print()
                print("The model is trained on the full development set.")
                print("The scores are computed on the full evaluation set.")
                print()
            y_true, y_pred = self.y_test, clf_results.predict(self.x_test)
            if self.debug:
                print(classification_report(y_true, y_pred))
                print()
        if self.debug:
            print(
                f"neural network {self.title} best_score: {clf_results.best_score_}\n, best_estimator: {clf_results.best_estimator_}\n, best_params: {clf_results.best_params_}"
            )
            print(
                f"neural network {self.title} test score: {clf_results.score(self.x_test, self.y_test)}"
            )

        if self.should_plot:
            plot_learning_curve(
                clf_results.best_estimator_,
                f"{self.title} Neural Network Classifier",
                self.x,
                self.y,
                filename=self.title,
                seed=self.seed,
                output_path=self.output_path,
            )


if __name__ == "__main__":
    AUSTRALIA = False
    AFRICA = True

    if AUSTRALIA:
        # AUSTRALIA DATA SET
        aus_X_train, aus_X_test, aus_y_train, aus_y_test = get_weather_data(
            10000, test_size=0.5
        )
        scaler = StandardScaler()
        scaler.fit(aus_X_train)
        aus_X_train = scaler.transform(aus_X_train)
        aus_X_test = scaler.transform(aus_X_test)

        aus_hyperparams = [
            {
                "hidden_layer_sizes": range(10, 30),
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
            title="Australian_Weather",
            seed=42,
            plot=True,
        )
        aus_tester.run()

    if AFRICA:
        # AFRICA DATA SET
        africa_X_train, africa_X_test, africa_y_train, africa_y_test = get_africa_data(
            test_size=0.2
        )
        scaler = StandardScaler()
        scaler.fit(africa_X_train)
        africa_X_train = scaler.transform(africa_X_train)
        africa_X_test = scaler.transform(africa_X_test)

        africa_hyperparams = [
            {
                "hidden_layer_sizes": range(10, 30),
                # 'max_iter': [100, 200, 300]
            }
        ]

        africa_tester = NNRunner(
            africa_X_train,
            africa_y_train,
            africa_X_test,
            africa_y_test,
            africa_hyperparams,
            title="Africa_Crisis",
            seed=42,
            plot=True,
        )
        africa_tester.run()
