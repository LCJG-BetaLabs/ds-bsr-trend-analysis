# Databricks notebook source
# COMMAND ----------

pip install minisom

# COMMAND ----------

import os
import math
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom


# COMMAND ----------

# result path
path = "/dbfs/mnt/dev/bsr_trend/clustering/clustering_result/"
os.makedirs(path, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # load data

# COMMAND ----------

df = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
df["order_week"] = pd.to_datetime(df["order_week"])
df["vpn"].nunique()

# COMMAND ----------

start, end = df["order_week"].min(), df["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# COMMAND ----------

# train_test_split
tr_start, tr_end = '2019-12-30', '2023-05-01'
te_start, te_end = '2023-05-08', '2023-11-06'
pred_start, pred_end = te_start, "2024-05-06"
real_pred_start = "2023-11-13"

# COMMAND ----------

vpns = np.unique(df["vpn"])

# fill 0 so that the length of each series is the same
my_series = []
for vpn in vpns:
    subdf = df[df["vpn"] == vpn].set_index("order_week")
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
    my_series.append(list(tra))

# COMMAND ----------

series_lengths = {len(series) for series in my_series}
print(series_lengths)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Self Organizing Maps (SOM)

# COMMAND ----------

som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(my_series))))
som = MiniSom(5, 5, len(my_series[0]), sigma=0.5, learning_rate = 0.05)
som.random_weights_init(my_series)
som.train(my_series, 50000)

# COMMAND ----------

winners = np.array([som.winner(x) for x in my_series])
win_map = som.win_map(my_series)

cluster_avg = []
# get cluster avg
for x in range(som_x):
    for y in range(som_y):
        cluster = (x, y)
        if cluster in win_map.keys():
            avg = np.average(np.vstack(win_map[cluster]), axis=0)
        cluster_number = x * som_y + y + 1
        cluster_avg.append((avg, cluster_number))

# COMMAND ----------

# save cluster averages
np.save(os.path.join(path, "cluster_averages.npy"), np.array(cluster_avg, dtype=object))

# COMMAND ----------

# MAGIC %md
# MAGIC ## visualize clustering result

# COMMAND ----------

# Little handy function to plot series
def plot_som_series_averaged_center(som_x, som_y, win_map):
    fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
    fig.suptitle('Clusters')
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                for series in win_map[cluster]:
                    axs[cluster].plot(series,c="gray",alpha=0.5) 
                axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
            cluster_number = x*som_y+y+1
            axs[cluster].set_title(f"Cluster {cluster_number}")

    plt.show()

# COMMAND ----------

win_map = som.win_map(my_series)
# Returns the mapping of the winner nodes and inputs
plot_som_series_averaged_center(som_x, som_y, win_map)

# COMMAND ----------

# Cluster Distribution
cluster_c = []
cluster_n = []
for x in range(som_x):
    for y in range(som_y):
        cluster = (x,y)
        if cluster in win_map.keys():
            cluster_c.append(len(win_map[cluster]))
        else:
            cluster_c.append(0)
        cluster_number = x*som_y+y+1
        cluster_n.append(f"Cluster {cluster_number}")

plt.figure(figsize=(25,5))
plt.title("Cluster Distribution for SOM")
plt.bar(cluster_n,cluster_c)
plt.show()

# COMMAND ----------

print(cluster_c)

# COMMAND ----------

def get_cluster_mapping_df(time_series, som, columns=["vpn", "cluster"]):
    cluster_map = []
    for idx in range(len(time_series)):
        winner_node = som.winner(time_series[idx])
        cluster_map.append((vpns[idx], winner_node[0] * som_y + winner_node[1] + 1))

    cluster_map = (
        pd.DataFrame(cluster_map, columns=columns)
        .sort_values(by="cluster")
        .reset_index(drop=True)
    )
    return cluster_map

# COMMAND ----------

cluster_mapping = get_cluster_mapping_df(my_series, som, columns=["vpn", "cluster"])
cluster_mapping.head()

# COMMAND ----------
# save result
cluster_mapping.to_csv(os.path.join(path, "cluster_mapping.csv"), index=False)

# save model
with open(os.path.join(path, "minisom_model.p"), 'wb') as outfile:
    pickle.dump(som, outfile)

# to load model
# with open(os.path.join(path, "minisom_model.p"), 'rb') as infile:
#     som = pickle.load(infile)

# COMMAND ----------


