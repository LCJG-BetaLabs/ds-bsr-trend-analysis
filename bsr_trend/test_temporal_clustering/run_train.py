# Databricks notebook source
import os
import pandas as pd
from multiprocessing.pool import ThreadPool

pool = ThreadPool(5)

# compare 3 clustering method
clustering_method = ["som", "kmeans", "kmeans_breakdown"]

for cm in clustering_method:
    path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
    mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
    pool.map(
        lambda cluster: dbutils.notebook.run(
            "./train",
            timeout_seconds=0,
            arguments={"cluster_no": cluster, "clustering_method": cm}),
        mapping["cluster"].values,
    )

# COMMAND ----------
