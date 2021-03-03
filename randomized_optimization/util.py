import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve


def get_weather_data(num_rows, test_size=0.5):
    df = pd.read_csv("data/weather-aus/weatherAUS_random.csv")
    df = df[:num_rows]

    categorical_cols = [
        "Location",
        "WindGustDir",
        "WindDir9am",
        "WindDir3pm",
        "RainToday",
        "RainTomorrow",
    ]
    excluded_columns = ["RISK_MM", "Date", "RainTomorrow", "RainToday"]
    # Fill all categorical columns nans with an applicable value
    for column_name in categorical_cols:
        df[column_name].fillna("None", inplace=True)

    # Fill all numerical columns (all that's left) nans with an applicable value
    df.fillna(0, inplace=True)

    # Give all discrete variables a numeric value
    le = preprocessing.LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

    y = df["RainTomorrow"]
    X = df.drop(excluded_columns, axis=1)
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)


def get_africa_data():
    df = pd.read_csv("data/africa-crisis/african_crises_shuffled.csv")

    categorical_cols = ["country"]
    binary_cols = ["banking_crisis"]

    # # Give binary variables a numeric value
    le = preprocessing.LabelEncoder()
    df[binary_cols] = df[binary_cols].apply(lambda col: le.fit_transform(col))

    # Turn categorical columns into dummy binary variables
    df = pd.get_dummies(df, columns=categorical_cols)

    y = df["systemic_crisis"]
    X = df.drop(
        [
            "systemic_crisis",
            "case",
            "cc3",
            "year",
            "currency_crises",
            "banking_crisis",
            "inflation_crises",
        ],
        axis=1,
    )
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


def plot_validation_curve(
    estimator,
    title,
    X_train,
    y_train,
    param_name,
    param_range,
    n_jobs=-1,
    semilog=False,
):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
    :param estimator: they type of estimator to use (eg. decision tree classifier)
    :param title: title of the graph
    :param X_train: X training data
    :param y_train: y target data for training
    :param param_name: name of parameter to vary
    :param param_range: range for parameter to vary over
    :return:
    """
    train_scores, test_scores = validation_curve(
        estimator,
        X_train,
        y_train,
        param_name=param_name,
        param_range=param_range,
        scoring="accuracy",
        n_jobs=n_jobs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Complexity Curve with {}".format(title))
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if not semilog:
        plt.plot(
            param_range, train_scores_mean, label="Training score", color="darkorange"
        )
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="darkorange",
        )
        plt.plot(
            param_range, test_scores_mean, label="Cross-validation score", color="navy"
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="navy",
        )
    else:
        plt.semilogx(
            param_range, train_scores_mean, label="Training score", color="darkorange"
        )
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="darkorange",
        )
        plt.semilogx(
            param_range, test_scores_mean, label="Cross-validation score", color="navy"
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="navy",
        )
    plt.legend(loc="best")
    return plt


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
