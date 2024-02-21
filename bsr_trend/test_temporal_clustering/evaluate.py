import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.metrics import dtw, cdist_dtw
from sklearn.metrics import silhouette_score as ss
from databricks.sdk.runtime import display


def silhouette_score(time_series: np.ndarray, labels: np.ndarray):
    return ss(time_series, labels, metric=dtw)


def average_dtw(time_series: np.ndarray, labels: np.ndarray):
    dtw_matrix = []
    for cluster in set(labels):
        subset = time_series[labels == cluster]
        # get dtw distance
        dtw = cdist_dtw(subset, subset)
        dtw_matrix.append(dtw)

    s = []
    for m in dtw_matrix:
        m = m[~np.eye(m.shape[0], dtype=bool)]  # exclude diagonal
        s.append((np.sum(m), np.mean(m), np.min(m) if m.shape[0] > 0 else 0, np.max(m) if m.shape[0] > 0 else 0, len(m)))

    return pd.DataFrame(s, columns=["total_dtw", "avg_dtw", "min_dtw", "max_dtw", "no_of_vpns"])


def plot_clusters(time_series: np.ndarray, labels: np.ndarray):
    plot_count = math.ceil(math.sqrt(len(np.unique(labels))))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(25, 25))
    fig.suptitle("Clusters")
    row_i = 0
    column_j = 0
    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
            if labels[i] == label:
                axs[row_i, column_j].plot(time_series[i], c="gray", alpha=0.4)
                cluster.append(time_series[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red")
        axs[row_i, column_j].set_title("Cluster " + str(label))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    display(plt.show())


def plot_cluster_distribution(labels: np.ndarray):
    cluster_c = [len(labels[labels == i]) for i in range(len(set(labels)))]
    cluster_n = ["Cluster " + str(i) for i in range(len(set(labels)))]
    print(cluster_c, cluster_n)

    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution")
    plt.bar(cluster_n, cluster_c)
    display(plt.show())
