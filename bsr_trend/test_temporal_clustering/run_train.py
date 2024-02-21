# Databricks notebook source
import os
import pandas as pd
import numpy as np

# compare 3 clustering method
clustering_method = ["som"]  # , "kmeans", "kmeans_breakdown"]

for cm in clustering_method:
    path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
    mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
    try:
        for cluster in mapping["cluster"].values:
            dbutils.notebook.run(
                "./train",
                timeout_seconds=0,
                arguments={"cluster_no": cluster, "clustering_method": cm},
            )
    except:
        print(cm, cluster)

# COMMAND ----------

cm = "kmeans"
path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
try:
    for cluster in mapping["cluster"].values:
        dbutils.notebook.run(
            "./train",
            timeout_seconds=0,
            arguments={"cluster_no": cluster, "clustering_method": cm},
        )
except:
    print(cm, cluster)

# COMMAND ----------

cm = "kmeans_breakdown"
path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
try:
    for cluster in mapping["cluster"].values:
        dbutils.notebook.run(
            "./train",
            timeout_seconds=0,
            arguments={"cluster_no": cluster, "clustering_method": cm},
        )
except:
    print(cm, cluster)

# COMMAND ----------


