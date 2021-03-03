import time

import mlrose_hiive as mlrose
from mlrose_hiive import GeomDecay
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import util

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = util.get_weather_data(10000, test_size=0.5)
    # https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    current_time = time.time()
    algorithms_to_run = [
        "simulated_annealing",
        "random_hill_climb",
        "genetic_alg",
        "gradient_descent",
    ]

    algorithm = "gradient_descent"
    if algorithm in algorithms_to_run:
        print(f"Running {algorithm}")
        nn = mlrose.NeuralNetwork(
            hidden_nodes=[29],
            activation="relu",
            algorithm=algorithm,
            max_iters=64,
            learning_rate=0.001,
            early_stopping=True,
            curve=True,
            random_state=42,
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        title = "Neural Network Classifier: {}".format(algorithm)
        util.plot_learning_curve(nn, title, X_train, y_train)
        print(classification_report(y_test, y_pred))
        plt.savefig("NN_{}_{}.png".format(algorithm, current_time))
        plt.close()

    algorithm = "random_hill_climb"
    """
    {'activation': <function relu at 0x10d1e0d30>, 'hidden_layer_sizes': [29], 'learning_rate': 1e-05, 'max_iters': 1024, 'restarts': 1}
    """
    if algorithm in algorithms_to_run:
        print(f"Running {algorithm}")
        nn = mlrose.NeuralNetwork(
            hidden_nodes=[29],
            activation="relu",
            algorithm=algorithm,
            max_iters=128,
            max_attempts=30,
            learning_rate=0.001,
            early_stopping=True,
            curve=True,
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        title = "Neural Network Classifier: {}".format(algorithm)
        util.plot_learning_curve(nn, title, X_train, y_train)
        print(classification_report(y_test, y_pred))
        plt.savefig("NN_{}_{}.png".format(algorithm, current_time))
        plt.close()

    algorithm = "simulated_annealing"
    if algorithm in algorithms_to_run:
        print(f"Running {algorithm}")
        nn = mlrose.NeuralNetwork(
            hidden_nodes=[29],
            activation="relu",
            algorithm=algorithm,
            max_iters=128,
            max_attempts=30,
            learning_rate=0.0001,
            early_stopping=True,
            curve=True,
            schedule=GeomDecay(init_temp=0.001, decay=0.95),
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        title = "Neural Network Classifier: {}".format(algorithm)
        util.plot_learning_curve(nn, title, X_train, y_train)
        print(classification_report(y_test, y_pred))
        plt.savefig("NN_{}_{}.png".format(algorithm, current_time))
        plt.close()

    algorithm = "genetic_alg"
    if algorithm in algorithms_to_run:
        print(f"Running {algorithm}")
        """
        {'activation': <function relu at 0x10ec3eca0>, 'hidden_layer_sizes': [29], 'learning_rate': 1e-05, 'max_iters': 100, 'mutation_prob': 0.1, 'pop_size': 40}
        """
        nn = mlrose.NeuralNetwork(
            hidden_nodes=[29],
            activation="relu",
            algorithm=algorithm,
            max_iters=50,
            pop_size=40,
            learning_rate=0.001,
            early_stopping=True,
            curve=True,
            random_state=42,
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        title = "Neural Network Classifier: {}".format(algorithm)
        util.plot_learning_curve(nn, title, X_train, y_train)
        print(classification_report(y_test, y_pred))
        plt.savefig("NN_{}_{}.png".format(algorithm, current_time))
        plt.close()

"""
Tuning data from A1 neural network classifier
neural network weather best_score: 0.7143825592087755
, best_estimator: MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=29, learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=100,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
, best_params: {'hidden_layer_sizes': 29, 'max_iter': 100}
neural network weather test score: 0.7025448081765664
"""
