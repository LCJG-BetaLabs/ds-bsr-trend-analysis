# Databricks notebook source
# MAGIC %pip install tslearn

# COMMAND ----------

# MAGIC %pip install fastdtw

# COMMAND ----------

import os
import math

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw

from bsr_trend.temporal_clustering.evaluate import silhouette_score, average_dtw, plot_clusters, plot_cluster_distribution

# COMMAND ----------

path = "/dbfs/mnt/dev/bsr_trend/clustering/kmeans/"
breakdown_path = "/dbfs/mnt/dev/bsr_trend/clustering/kmeans_breakdown/"
os.makedirs(path, exist_ok=True)
os.makedirs(breakdown_path, exist_ok=True)

# COMMAND ----------

# load data
sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
sales["order_week"] = pd.to_datetime(sales["order_week"])

vpn_info = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_info.csv")

# COMMAND ----------

sales = sales.merge(vpn_info[["vpn", "category"]], how="left", on="vpn")
# clustering by category
# dev: take spa for testing
sales = sales[sales["category"] == '6409- Home Fragrance & Spa']
sales["vpn"].nunique()

# COMMAND ----------

start, end = sales["order_week"].min(), sales["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# train_test_split
tr_start, tr_end = start.strftime('%Y-%m-%d'), '2023-09-01'
te_start, te_end = '2023-09-01', '2023-11-30'
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]
pred_buf = pred_buf.set_index("order_week")

# COMMAND ----------

vpns = np.unique(sales["vpn"])

# fill 0 so that the length of each series is the same
time_series = []
for vpn in vpns:
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
    buf = pd.merge(
        pd.DataFrame(index=date_range),
        subdf,
        how="left",
        left_index=True,
        right_index=True,
    )
    buf["order_week"] = buf.index
    buf = buf.drop(["vpn", "amt", "order_week", "category"], axis=1)
    buf = buf.fillna(0).astype(int)

    # take training period
    tra = buf['qty'][tr_start:tr_end].dropna()
    tra.sort_index(inplace=True)
    time_series.append(list(tra))

time_series = np.array(time_series)
time_series.shape

# COMMAND ----------

# Dynamic Time Warping (DTW) + k-mean clustering
cluster_count = math.ceil(math.sqrt(len(time_series))) 
print(cluster_count)
# A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", random_state=0)
labels = km.fit_predict(time_series)

# COMMAND ----------

silhouette_score(time_series, labels)

# COMMAND ----------

display(average_dtw(time_series, labels))

# COMMAND ----------

plot_clusters(time_series, labels)

# COMMAND ----------

cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
cluster_n = ["Cluster " + str(i) for i in range(cluster_count)]
print(cluster_c, cluster_n)

plot_cluster_distribution(labels)

# COMMAND ----------

# save result
result = pd.DataFrame(zip(vpns, labels), columns=["vpn", "cluster"])
result.to_csv(os.path.join(path, "cluster_mapping.csv"), index=False)

# save model
with open(os.path.join(path, "kmean_dtw_model.pkl"), "wb") as f:
    pickle.dump(km, f)

# COMMAND ----------

# MAGIC %md
# MAGIC cluster breakdown

# COMMAND ----------

# further cluster the big clusters (0)
big_cluster = [0]

vpns_list = []
models_list = []
labels_list = []

for c in big_cluster:
    tmp = result[result["cluster"] == c]
    _vpns = np.unique(tmp["vpn"])

    time_series = []
    for vpn in _vpns:
        subdf = sales[sales["vpn"] == vpn].set_index("order_week")
        buf = pd.merge(
            pd.DataFrame(index=date_range),
            subdf,
            how="left",
            left_index=True,
            right_index=True,
        )
        buf["order_week"] = buf.index
        buf = buf.drop(["vpn", "amt", "order_week", "category"], axis=1)
        buf = buf.fillna(0).astype(int)

        # take training period
        tra = buf['qty'][tr_start:tr_end].dropna()
        tra.sort_index(inplace=True)
        time_series.append(list(tra))

    cluster_count = math.ceil(math.sqrt(len(time_series))) 
    print(cluster_count)
    # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

    km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", random_state=0)
    labels = km.fit_predict(time_series)

    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(time_series))))

    plot_count = math.ceil(math.sqrt(cluster_count))
    
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
        axs[row_i, column_j].set_title("Cluster " + str(row_i * som_y + column_j))
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0

    display(plt.show())
    vpns_list.append(_vpns)
    models_list.append(km)
    labels_list.append(labels)

# COMMAND ----------

vpns_list

# COMMAND ----------

len(time_series)

# COMMAND ----------

labels_list

# COMMAND ----------

from tslearn.metrics import dtw

# COMMAND ----------

silhouette_score(time_series, labels_list[0], metric=dtw)

# COMMAND ----------

labels = labels_list[0]

# COMMAND ----------

dtw_matrixs = []
for cluster in set(labels):
    print(f"Cluster {cluster}")
    print(f"Cluster size: {sum(labels == cluster)}")
    subset = np.array(time_series)[labels == cluster]
    # get dtw distance
    dtw = cdist_dtw(subset, subset)
    print(dtw)
    dtw_matrixs.append(dtw)

s = []
for m in dtw_matrixs:
    print(np.sum(m), np.max(m), np.mean(m), np.mean(m[~np.eye(m.shape[0], dtype=bool)]))
    s.append(np.mean(m[~np.eye(m.shape[0], dtype=bool)]))

display(pd.DataFrame(zip(s, [len(m) for m in dtw_matrixs])))

# COMMAND ----------

for _v, l, num in zip(vpns_list, labels_list, big_cluster):
    df = pd.DataFrame(zip(_v, l), columns=["vpn", "cluster"])
    # df.to_csv(os.path.join(path, f"subcluster_mapping_cluster_{num}.csv"), index=False)

# COMMAND ----------

for l in labels_list:
    print(np.unique(l, return_counts=True))

# COMMAND ----------



# COMMAND ----------

df

# COMMAND ----------

result

# COMMAND ----------

labels = result["cluster"].values

# COMMAND ----------

labels

# COMMAND ----------

labels[labels != 0] = labels[labels != 0] + 12

# COMMAND ----------

labels[labels == 0] = labels_list[0]

# COMMAND ----------

from tslearn.metrics import dtw

silhouette_score(time_series, labels, metric=dtw)

# COMMAND ----------

final_result = pd.DataFrame(zip(vpns, labels), columns=["vpn", "cluster"])

# COMMAND ----------

final_result["cluster"].unique()

# COMMAND ----------

# save result
final_result.to_csv(os.path.join(breakdown_path, "cluster_mapping.csv"), index=False)

# save model
with open(os.path.join(breakdown_path, "kmean_dtw_model.pkl"), "wb") as f:
    pickle.dump(km, f)

# COMMAND ----------

os.path.join(breakdown_path, "cluster_mapping.csv")

# COMMAND ----------


