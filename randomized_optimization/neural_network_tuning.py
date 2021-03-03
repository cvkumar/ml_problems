import mlrose_hiive as mlrose
from mlrose_hiive import GeomDecay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from mlrose_hiive.runners import NNGSRunner
import matplotlib.pyplot as plt

import util


def tune_nn_gradient_descent():
    X_train, X_test, y_train, y_test = util.get_weather_data(10000)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search_parameters = {
        "max_iters": [64, 100, 128],  # nn params
        "learning_rate": [0.0001, 0.001, 0.002, 0.01],  # nn params
        # 'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    }

    nnr = NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        experiment_name="nn_test",
        algorithm=mlrose.algorithms.gradient_descent,
        grid_search_parameters=grid_search_parameters,
        iteration_list=[100, 500, 1000, 2500],
        hidden_layer_sizes=[[29]],
        bias=True,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=30,
        generate_curves=True,
        seed=42,
        activation=[mlrose.neural.activation.relu],
        output_directory="../",
    )

    (
        run_stats_df,
        curves_df,
        cv_results_df,
        sr,
    ) = nnr.run()  # GridSearchCV instance returned
    print(sr)


def tune_nn_sa():
    X_train, X_test, y_train, y_test = util.get_weather_data(10000)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search_parameters = {
        "max_iters": [64, 128],  # nn params
        "learning_rate": [0.0001, 0.001, 0.002, 0.01],  # nn params
        "schedule": [
            GeomDecay(init_temp=1, decay=0.95),
            GeomDecay(init_temp=0.1, decay=0.95),
            GeomDecay(init_temp=10, decay=0.95),
        ],  # sa params
    }

    nnr = NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        experiment_name="nn_test",
        algorithm=mlrose.algorithms.sa.simulated_annealing,
        grid_search_parameters=grid_search_parameters,
        iteration_list=[500],
        hidden_layer_sizes=[[29]],
        bias=True,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=30,
        generate_curves=True,
        seed=42,
        activation=[mlrose.neural.activation.relu],
        output_directory="../",
        cv=3,
    )

    (
        run_stats_df,
        curves_df,
        cv_results_df,
        sr,
    ) = nnr.run()  # GridSearchCV instance returned
    print(sr.best_params_)


def tune_nn_rhc():
    X_train, X_test, y_train, y_test = util.get_weather_data(10000)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search_parameters = {
        "max_iters": [128, 256, 512, 1024],  # nn params
        "learning_rate": [0.00001, 0.0001, 0.001, 0.002],
        "restarts": [1]
        # nn params
        # 'schedule': [ArithDecay(1), ArithDecay(100), ArithDecay(1000)]  # sa params
    }

    nnr = NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        experiment_name="nn_test",
        algorithm=mlrose.algorithms.random_hill_climb,
        grid_search_parameters=grid_search_parameters,
        iteration_list=[500],
        hidden_layer_sizes=[[29]],
        bias=True,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=100,
        generate_curves=True,
        seed=42,
        activation=[mlrose.neural.activation.relu],
        output_directory="../",
    )

    (
        run_stats_df,
        curves_df,
        cv_results_df,
        sr,
    ) = nnr.run()  # GridSearchCV instance returned
    print(sr.best_params_)


def tune_nn_ga():
    X_train, X_test, y_train, y_test = util.get_weather_data(10000)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search_parameters = {
        "max_iters": [100],
        "learning_rate": [0.00001, 0.0001, 0.001, 0.002],
        "pop_size": [20, 40],
        "mutation_prob": [0.1, 0.2, 0.3, 0.4],
    }

    nnr = NNGSRunner(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        experiment_name="nn_test",
        algorithm=mlrose.algorithms.genetic_alg,
        grid_search_parameters=grid_search_parameters,
        iteration_list=[500],
        hidden_layer_sizes=[[29]],
        bias=True,
        early_stopping=True,
        clip_max=1e10,
        max_attempts=20,
        generate_curves=True,
        seed=42,
        activation=[mlrose.neural.activation.relu],
        output_directory="../",
    )

    (
        run_stats_df,
        curves_df,
        cv_results_df,
        sr,
    ) = nnr.run()  # GridSearchCV instance returned
    print(sr.best_params_)


if __name__ == "__main__":
    # tune_nn_gradient_descent()
    # tune_nn_gradient_descent()
    # tune_nn_sa()
    tune_nn_ga()
