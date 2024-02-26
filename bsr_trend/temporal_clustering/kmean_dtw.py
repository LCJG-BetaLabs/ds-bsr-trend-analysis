# Databricks notebook source
# MAGIC %pip install tslearn

# COMMAND ----------

# MAGIC %pip install fastdtw

# COMMAND ----------

import os
import math

import pickle
import pandas as pd
import numpy as np

from tslearn.clustering import TimeSeriesKMeans

from bsr_trend.temporal_clustering.evaluate import (
    silhouette_score,
    average_dtw,
    plot_clusters,
    plot_cluster_distribution,
)
from bsr_trend.utils.data import (
    get_sales_table,
    get_time_series,
)
from bsr_trend.utils.catalog import (
    write_uc_table,
    CLUSTERING_MAPPING,
    KMEAN_DIRS,
    TREND_CLASSIFICATION_RESULT,
)

# COMMAND ----------


def kmean_dtw(time_series, vpns, train_end, save=True, evaluate=True, model_name=None):
    """Dynamic Time Warping (DTW) + k-mean clustering"""
    # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN
    cluster_count = math.ceil(math.sqrt(len(time_series)))
    print(f"number of cluster = {cluster_count}")

    km_model = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", random_state=0)
    labels = km_model.fit_predict(time_series)

    if save:
        result = pd.DataFrame({"vpn": vpns, "cluster": labels, "train_end": train_end})
        #, columns=["vpn", "cluster", "train_end"])
        result["cluster"] = result["cluster"].apply(lambda l: f"{model_name}_{l}")
        write_uc_table(
            CLUSTERING_MAPPING,
            result,
            mode="append",
        )

        # save model
        with open(os.path.join(KMEAN_DIRS, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(km_model, f)

    if evaluate:
        print(f"silhouette_score = {silhouette_score(time_series, labels)}")
        print("average_dtw", average_dtw(time_series, labels))
        plot_clusters(time_series, labels)

        cluster_c = [len(labels[labels == i]) for i in range(cluster_count)]
        cluster_n = ["Cluster " + str(i) for i in range(cluster_count)]
        print(cluster_c, cluster_n)

        plot_cluster_distribution(labels)

    return labels, km_model


# COMMAND ----------

# load data
sales = get_sales_table()
vpns = np.unique(sales["vpn"])
tr_end = "2023-09-01"

# COMMAND ----------

# smooth and irregular
trend = spark.sql(
    f"""
    SELECT * FROM {TREND_CLASSIFICATION_RESULT}
    WHERE trend_class IN ("smooth", "irregular")
    AND history != "history<3months"
    AND train_end = "{tr_end}"
""")
display(trend)

# COMMAND ----------

sales = sales.merge(trend.select("vpn", "trend_class").toPandas(), on="vpn", how="inner")

# COMMAND ----------

for tc in np.unique(sales["trend_class"]):
    subdf = sales[sales["trend_class"] == tc].drop(columns="trend_class")
    time_series = get_time_series(subdf, dynamic_start=False, end_date=tr_end)
    time_series = np.array(time_series)
    print(time_series.shape)

    labels, km = kmean_dtw(time_series, np.unique(subdf["vpn"]), tr_end, save=True, evaluate=False, model_name=f"kmean_{tr_end}_{tc}_spa")
