from sklearn.cluster import KMeans
from typing import Optional

from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
from sklearn import metrics, decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import graph_with_num_clusters, plot_clustering_labels_histogram


class KMeansRunner:
    def __init__(
        self,
        x,
        y,
        clusters: list,
        seed=1,
        should_plot=False,
        write_labels=False,
        title="",
        chosen_cluster=None,
        n_init=10,
        max_iters=500,
        compute_silhouette=False,
        result_path="",
        histogram=False,
    ):
        self.x = x
        self.y = y
        self.clusters = clusters
        self.should_plot = should_plot
        self.seed = seed
        self.title = title
        self.chosen_cluster = chosen_cluster
        self.write_labels = write_labels
        self.result_path = result_path

        self.n_init = n_init
        self.max_iters = max_iters
        self.histogram = histogram

        # Metrics
        self.sum_of_squared_errors = []
        self.homogeneity_scores = []
        self.completeness_scores = []
        self.rand_scores = []
        self.v_measure = []
        self.compute_silhouette = compute_silhouette
        self.silhouette_scores = []

    def _write_labels_csv(self, labels):
        nd_data = np.concatenate(
            (self.x, np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1)),
            axis=1,
        )
        pd_data = pd.DataFrame(nd_data)
        pd_data.to_csv(
            f"{self.result_path}{self.title}_cluster_kmeans_labels.csv",
            index=False,
            index_label=False,
            header=False,
        )

    def run(self):

        for k in self.clusters:
            kmeans = KMeans(
                n_clusters=k,
                max_iter=self.max_iters,
                init="k-means++",
                n_jobs=-1,
                n_init=self.n_init,
                random_state=self.seed,
            )
            labels = kmeans.fit_predict(self.x)
            if k == self.chosen_cluster and self.write_labels:
                # self._write_labels_csv(labels)
                if self.histogram:
                    plot_clustering_labels_histogram(
                        title="KM Clustering Histogram",
                        labels=labels,
                        output_path=self.result_path,
                    )

            self.sum_of_squared_errors.append(kmeans.inertia_)

            """
            A clustering result satisfies homogeneity if all of its clusters contain only data points which are members 
            of a single class.

            A clustering result satisfies completeness if all the data points that are members of a given class are 
            elements of the same cluster.

            Both scores have positive values between 0.0 and 1.0, larger values being desirable.
            """
            self.homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            self.completeness_scores.append(metrics.completeness_score(self.y, labels))

            """
            The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples 
            and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
            
            The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently of 
            the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation).
            """
            self.rand_scores.append(metrics.adjusted_rand_score(self.y, labels))
            if self.compute_silhouette:
                self.silhouette_scores.append(metrics.silhouette_score(self.x, labels))

        if self.should_plot:
            self._plot()

    def _plot(self):
        graph_with_num_clusters(
            title=self.title,
            y_label="Sum of Squared Errors",
            y_data=self.sum_of_squared_errors,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
            color="b",
        )

        if self.compute_silhouette:
            graph_with_num_clusters(
                title=self.title + " silhouette score",
                y_label="Silhouette scores",
                y_data=self.silhouette_scores,
                x_data=self.clusters,
                seed=self.seed,
                color="b",
                path=self.result_path,
            )

        graph_with_num_clusters(
            title=self.title + " homogeneity score",
            y_label="homogeneity score",
            y_data=self.homogeneity_scores,
            x_data=self.clusters,
            seed=self.seed,
            color="b",
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + " completeness score",
            y_label="completeness score",
            y_data=self.completeness_scores,
            x_data=self.clusters,
            seed=self.seed,
            color="b",
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + " adjusted rand score",
            y_label="adjusted rand score",
            y_data=self.rand_scores,
            x_data=self.clusters,
            seed=self.seed,
            color="b",
            path=self.result_path,
        )
