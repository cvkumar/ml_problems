import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import util


def neural_network_weather():
    X_train, X_test, y_train, y_test = util.get_weather_data(10000)

    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # generate validation scores over a range of param values
    util.plot_validation_curve(
        MLPClassifier(),
        "Neural Network Classifier Dataset 1",
        X_train,
        y_train,
        "hidden_layer_sizes",
        range(1, 100),
    )
    plt.show()

    clf_parameters = [
        {
            "hidden_layer_sizes": range(10, 30),
            # 'activation': ['tanh', 'relu'],
            # 'solver': ['sgd', 'adam'],
            # 'alpha': [0.0001, 0.05],
            # 'learning_rate': ['constant', 'adaptive'],
            "max_iter": [1, 10, 100],
        }
    ]

    scores = ["precision", "recall"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(MLPClassifier(), clf_parameters, scoring="%s_macro" % score)
        clf_results = clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf_results.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_results.cv_results_["mean_test_score"]
        stds = clf_results.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf_results.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf_results.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    print(
        f"neural network weather best_score: {clf_results.best_score_}\n, best_estimator: {clf_results.best_estimator_}\n, best_params: {clf_results.best_params_}"
    )
    print(f"neural network weather test score: {clf_results.score(X_test, y_test)}")

    util.plot_learning_curve(
        clf_results.best_estimator_,
        "Neural Network Classifier Dataset 1",
        X_train,
        y_train,
    )
    # plt.show()


def neural_network_africa():
    X_train, X_test, y_train, y_test = util.get_africa_data()

    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # generate validation scores over a range of param values
    util.plot_validation_curve(
        MLPClassifier(),
        "Neural Network Classifier Dataset 2",
        X_train,
        y_train,
        "hidden_layer_sizes",
        range(1, 100),
    )
    plt.show()

    clf_parameters = [
        {
            "hidden_layer_sizes": range(10, 30),
            # 'max_iter': [100, 200, 300]
        }
    ]

    scores = ["precision", "recall"]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(MLPClassifier(), clf_parameters, scoring="%s_macro" % score)
        clf_results = clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf_results.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_results.cv_results_["mean_test_score"]
        stds = clf_results.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf_results.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf_results.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    print(
        f"neural network africa best_score: {clf_results.best_score_}\n, best_estimator: {clf_results.best_estimator_}\n, best_params: {clf_results.best_params_}"
    )
    print(f"neural network africa test score: {clf_results.score(X_test, y_test)}")

    util.plot_learning_curve(
        clf_results.best_estimator_,
        "Neural Network Classifier Dataset 2",
        X_train,
        y_train,
    )
    # plt.show()


if __name__ == "__main__":
    neural_network_weather()
    # neural_network_africa()
