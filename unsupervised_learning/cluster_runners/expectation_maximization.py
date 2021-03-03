from typing import Optional

from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
from sklearn import metrics, decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asgn3_unsupervised_learning.common import graph_with_num_clusters


class ExpectationMaximizationRunner:
    def __init__(
        self,
        x,
        y,
        clusters: list,
        seed=1,
        plot=False,
        write_labels=False,
        title="",
        chosen_cluster=None,
        compute_silhouette=False,
        result_path="",
    ):
        self.x = x
        self.y = y
        self.clusters = clusters
        self.should_plot = plot
        self.seed = seed
        self.title = title
        self.chosen_cluster = chosen_cluster
        self.write_labels = write_labels
        self.result_path = result_path

        # Metrics
        self.log_likelihood_scores = []
        self.homogeneity_scores = []
        self.completeness_scores = []
        self.rand_scores = []
        self.bic = []
        self.aic = []

        self.compute_silhouette = compute_silhouette
        self.silhouette_scores = []

    def _save_labels(self, labels):
        nd_data = np.concatenate(
            (self.x, np.expand_dims(labels, axis=1), np.expand_dims(self.y, axis=1)),
            axis=1,
        )
        pd_data = pd.DataFrame(nd_data)
        pd_data.to_csv(
            f"{self.result_path}{self.title}_cluster_em.csv",
            index=False,
            index_label=False,
            header=False,
        )
        for i in range(0, self.chosen_cluster):
            cluster = pd_data.loc[pd_data.iloc[:, -2] == i].iloc[:, -2:]
            cluster.shape[0]

    def run(self):
        em = GaussianMixture(covariance_type="diag", random_state=self.seed)

        for k in self.clusters:
            em.set_params(n_components=k)
            em.fit(self.x)
            labels = em.predict(self.x)
            if self.chosen_cluster and self.write_labels and self.chosen_cluster == k:
                self._save_labels(labels)
            em.score(self.x)

            # https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4

            self.homogeneity_scores.append(metrics.homogeneity_score(self.y, labels))
            self.completeness_scores.append(metrics.completeness_score(self.y, labels))
            self.rand_scores.append(metrics.adjusted_rand_score(self.y, labels))

            """
            The lower is the BIC, the better is the model to actually predict the data we have, and by extension, the true, unknown, distribution. 
            """
            self.bic.append(em.bic(self.x))
            self.aic.append(em.aic(self.x))
            self.log_likelihood_scores.append(em.score(self.x))

            """
            The mean distance between a sample and all other points in the same cluster.
            The mean distance between a sample and all other points in the next nearest cluster.
            i.e. it checks how much the clusters are compact and well separated. The more the score is near to one, the better the clustering is.
            """
            if self.compute_silhouette:
                self.silhouette_scores.append(metrics.silhouette_score(self.x, labels))

        if self.should_plot:
            self._plot()

    def _plot(self):
        graph_with_num_clusters(
            title=self.title + "_Log_Likelihood",
            y_label="Log Likelihood",
            y_data=self.log_likelihood_scores,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + "_BIC",
            y_label="BIC",
            y_data=self.bic,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + "_AIC",
            y_label="AIC",
            y_data=self.aic,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + "_RAND",
            y_label="RAND SCORE",
            y_data=self.rand_scores,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + "_Homogeneity_Score",
            y_label="Homogeneity Score",
            y_data=self.homogeneity_scores,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        graph_with_num_clusters(
            title=self.title + "_Completeness_Score",
            y_label="Completeness Score",
            y_data=self.completeness_scores,
            x_data=self.clusters,
            seed=self.seed,
            path=self.result_path,
        )

        if self.compute_silhouette:
            graph_with_num_clusters(
                title=self.title + "_Silhouette_Score",
                y_label="Silhouette Score",
                y_data=self.silhouette_scores,
                x_data=self.clusters,
                seed=self.seed,
                path=self.result_path,
            )
