"""
3) Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
Yes, that’s 16 combinations of datasets, dimensionality reduction, and clustering method.
You should look at all of them, but focus on the more interesting findings in your report.
- Run dimensionality reduction algorithms on the data, then run your clustering algorithm again on the result
- 2 (datasets) * 2 (Clustering) * 4 (Dimensionality) = 16 results
- Focus on interesting findings
"""
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection

from expectation_maximization import ExpectationMaximizationRunner
from kmeans import KMeansRunner
from util import get_weather_data, get_africa_data
import matplotlib.pyplot as plt


def get_ica_data(x, n_components):
    """
    Australia components = 9
    Africa components = 14
    """
    ica = FastICA(n_components=n_components)
    return ica.fit_transform(x)


def get_pca_data(x, n_components):
    """
    Australia components = 12
    Africa components = 14
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)


def get_grp_data(x, n_components):
    """
    Australia components = 10
    Africa components = 10
    """
    transformed_x = []
    # iterate 20 times
    for i in range(20):
        grp = GaussianRandomProjection(random_state=i, n_components=n_components)
        transformed_x.append(grp.fit_transform(x))
    return np.mean(transformed_x, axis=0)


def get_rf_data(x, components):
    """
    Australia components = [3, 7, 12, 13, 14, 15]
    Africa components = [0, 4]
    """
    return x.iloc[:, components]


def get_weather_plots(seed=0):
    X, y = get_weather_data(10000, split_train_test=False)
    ica = get_ica_data(X, 9), "ica"
    pca = get_pca_data(X, 12), "pca"
    grp = get_grp_data(X, 10), "grp"
    rf = get_rf_data(X, [3, 7, 12, 13, 14, 15]), "rf"
    original = X, "original"
    reduced_datasets = [original, ica, pca, grp, rf]
    em_silhouette_scores = []
    kmeans_sse_scores = []

    for x_dataset in reduced_datasets:
        reduction_name = x_dataset[1]
        X = x_dataset[0]
        print(f"Running Weather Reduced Clustering for {reduction_name}")
        em = ExpectationMaximizationRunner(
            X,
            y,
            title=f"Australian_Weather_EM_Reduced_{reduction_name}",
            seed=seed,
            clusters=list(range(2, 20)),
            plot=True,
            chosen_cluster=4,
            write_labels=True,
            compute_silhouette=True,
            result_path=f"results/reduction_clustering_weather/{reduction_name}/em/",
            histogram=True,
        )
        em.run()
        em_silhouette_scores.append(em.silhouette_scores)

        kmeans = KMeansRunner(
            X,
            y,
            list(range(2, 20)),
            should_plot=True,
            write_labels=True,
            title=f"Australian_Weather_Kmeans_Reduced_{reduction_name}",
            chosen_cluster=4,
            n_init=10,
            max_iters=500,
            seed=seed,
            compute_silhouette=False,
            result_path=f"results/reduction_clustering_weather/{reduction_name}/km/",
            histogram=True,
        )
        kmeans.run()
        kmeans_sse_scores.append(kmeans.sum_of_squared_errors)

    plot_scores(
        ys=em_silhouette_scores,
        ylabel="Silhouette Score",
        num_clusters=em.clusters,
        title=f"Weather Reduced Clustering Silhouette Scores",
        seed=seed,
        output_path=f"results/reduction_clustering_weather/",
    )

    plot_scores(
        ys=kmeans_sse_scores,
        ylabel="Sum of Squared Errors",
        num_clusters=kmeans.clusters,
        title=f"Weather Reduced Clustering SSE",
        seed=seed,
        output_path=f"results/reduction_clustering_weather/",
    )


def get_africa_plots(seed=0):
    X, y = get_africa_data(split_train_test=False)
    ica = get_ica_data(X, 14), "ica"
    pca = get_pca_data(X, 14), "pca"
    grp = get_grp_data(X, 10), "grp"
    rf = get_rf_data(X, [0, 4]), "rf"
    original = X, "original"
    reduced_datasets = [original, ica, pca, grp, rf]
    em_silhouette_scores = []
    kmeans_sse_scores = []

    for x_dataset in reduced_datasets:
        reduction_name = x_dataset[1]
        X = x_dataset[0]
        print(f"Running Africa Reduced Data Clustering for {reduction_name}")
        em = ExpectationMaximizationRunner(
            X,
            y,
            title=f"Africa_EM_Reduced_{reduction_name}",
            seed=seed,
            clusters=list(range(2, 20)),
            plot=True,
            chosen_cluster=4,
            write_labels=False,
            compute_silhouette=True,
            result_path=f"results/reduction_clustering_africa/{reduction_name}/em/",
        )
        em.run()
        em_silhouette_scores.append(em.silhouette_scores)

        kmeans = KMeansRunner(
            X,
            y,
            list(range(3, 20)),
            should_plot=True,
            write_labels=False,
            title=f"Africa_Kmeans_Reduced_{reduction_name}",
            chosen_cluster=4,
            n_init=10,
            max_iters=500,
            seed=seed,
            compute_silhouette=False,
            result_path=f"results/reduction_clustering_africa/{reduction_name}/km/",
        )
        kmeans.run()
        kmeans_sse_scores.append(kmeans.sum_of_squared_errors)

    plot_scores(
        ys=em_silhouette_scores,
        ylabel="Silhouette Score",
        num_clusters=em.clusters,
        title=f"Africa Reduced Clustering Silhouette Scores",
        seed=seed,
        output_path=f"results/reduction_clustering_africa/",
    )

    plot_scores(
        ys=kmeans_sse_scores,
        ylabel="Sum of Squared Errors",
        num_clusters=kmeans.clusters,
        title=f"Africa Reduced Clustering SSE",
        seed=seed,
        output_path=f"results/reduction_clustering_africa/",
    )


def plot_scores(ys, ylabel, num_clusters, title, seed=42, output_path=""):
    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()

    for i in range(len(ys)):
        y = ys[i]
        plt.plot(num_clusters, y, marker="o", linestyle="--")
    plt.legend(["Original", "ICA", "PCA", "GRP", "RF"], loc="upper right")

    plt.xlabel("Number of Clusters")
    plt.xticks(
        np.arange(0, len(num_clusters) + 3, step=1)
    )  # change from 0-based array index to 1-based human-readable label
    plt.ylabel(ylabel)
    plt.title(f"{title}")

    ax.grid(axis="x")

    file_name = f"{output_path}{title}_{seed}.png"
    plt.savefig("{}".format(file_name))
    plt.close()


if __name__ == "__main__":
    get_africa_plots()
    get_weather_plots()
