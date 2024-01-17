# Databricks notebook source
# MAGIC %pip install tslearn

# COMMAND ----------

import os
import math

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans

from sklearn.decomposition import PCA

# COMMAND ----------

path = "/dbfs/mnt/dev/bsr_trend/clustering/dtw_clustering_result/"
os.makedirs(path, exist_ok=True)

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
tr_start = '2019-12-30'
tr_end = '2023-09-01'
te_start, te_end = '2023-09-01', '2023-11-30'
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]
pred_buf = pred_buf.set_index("order_week")

# COMMAND ----------

# # dynamic start date: based on first purchase of the vpn
# vpns = np.unique(sales["vpn"])

# time_series = []

# for vpn in vpns:
#     subdf = sales[sales["vpn"] == vpn].set_index("order_week")
#     buf = pd.merge(
#         pd.DataFrame(index=date_range),
#         subdf,
#         how="left",
#         left_index=True,
#         right_index=True,
#     )
#     buf["order_week"] = buf.index
#     tr_start = buf[buf.vpn.notna()]["order_week"].min().to_pydatetime()

#     buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
#     buf = buf.fillna(0).astype(int)

#     # take training period
#     tra = buf["qty"][tr_start:tr_end].dropna()
#     tra.sort_index(inplace=True)
#     time_series.append(list(tra))

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
    buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
    buf = buf.fillna(0).astype(int)

    # take training period
    tra = buf['qty'][tr_start:tr_end].dropna()
    tra.sort_index(inplace=True)
    time_series.append(list(tra))

# COMMAND ----------

len(time_series)

# COMMAND ----------

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

som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(time_series))))

# COMMAND ----------

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

# COMMAND ----------

cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
cluster_n = ["Cluster " + str(i) for i in range(cluster_count)]
print(cluster_c, cluster_n)

plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()

# COMMAND ----------

result = pd.DataFrame(zip(vpns, labels), columns=["vpns", "cluster"])
result

# COMMAND ----------

# save result
result.to_csv(os.path.join(path, "cluster_mapping.csv"), index=False)

# save model
with open(os.path.join(path, "kmean_dtw_model.pkl"), "wb") as f:
    pickle.dump(km, f)

# COMMAND ----------


