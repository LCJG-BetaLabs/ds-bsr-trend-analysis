# Databricks notebook source
import base64
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from pmdarima.arima import auto_arima

from bsr_trend.logger import get_logger
from bsr_trend.utils.data import get_sales_table, get_time_series
from bsr_trend.utils.catalog import CLUSTERING_MAPPING

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")
logger = get_logger()

path = f"/dbfs/mnt/dev/bsr_trend/clustering/kmeans_dtw/"
result_path = os.path.join(path, "arima_result")
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

# load data
sales = get_sales_table()
cluster_mapping = spark.table(CLUSTERING_MAPPING).toPandas()
sales = sales.merge(cluster_mapping[["vpn", "cluster"]], on="vpn", how="left")
sales = sales[~sales["cluster"].isna()]

# COMMAND ----------

start, end = sales["order_week"].min(), sales["order_week"].max()

# train_test_split
tr_start, tr_end = start.strftime("%Y-%m-%d"), "2023-09-01"
te_start, te_end = "2023-09-01", "2023-11-30"
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

# COMMAND ----------

distinct_cluster = np.unique(sales["cluster"])
report = []
for cluster in tqdm(distinct_cluster):
    subdf = sales[sales["cluster"] == cluster]
    tra = get_time_series(subdf, dynamic_start=True, start_date=None, end_date=tr_end)
    tes = get_time_series(subdf, dynamic_start=False, start_date=te_start, end_date=te_end)

    save_path = os.path.join(result_path, cluster)
    os.makedirs(save_path, exist_ok=True)

    vpns = np.unique(subdf["vpn"])
    gt_and_pred = []

    for vpn, _tra, _tes in zip(vpns, tra, tes):
        model = auto_arima(_tra, seasonal=True, m=52, trace=True)
        print(model.summary())

        # save model
        encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
        folder = os.path.join(save_path, encoded_vpn)
        os.makedirs(folder, exist_ok=True)
        sm.iolib.smpickle.save_pickle(model, f"{folder}/arima.pkl")
        # save model summary
        summary = model.summary()
        with open(f"{folder}/model_summary.txt", 'w') as file:
            file.write(str(summary))

        predictions = model.predict(n_periods=12)
        test_and_pred = [vpn, sum(_tes), sum(predictions)]
        gt_and_pred.append(test_and_pred)

    agg_testing_error = pd.DataFrame(gt_and_pred, columns=["vpn", "gt", "model_pred"])
    agg_testing_error["model_mape (%)"] = abs(agg_testing_error["model_pred"] / (agg_testing_error["gt"]) - 1) * 100

    # get sales vel
    sales_velocities = {}
    for vpn in tqdm(vpns, total=len(vpns)):
        subdf = sales[sales["vpn"] == vpn].set_index("order_week")
        start, end = subdf.index.min(), subdf.index.max()
        date_range = pd.date_range(start, end, freq="W-MON")
        buf = pd.merge(
            pd.DataFrame(index=date_range),
            subdf,
            how="left",
            left_index=True,
            right_index=True,
        )
        buf["order_week"] = buf.index
        buf["qty"] = buf["qty"].fillna(0).astype(int)
        buf = buf[["order_week", "qty"]]
        buf = buf[(buf["order_week"] >= tr_start) & (buf["order_week"] <= tr_end)]
        sales_velocity = buf["qty"].mean()
        sales_velocities[vpn] = sales_velocity

    sales_velocities = pd.DataFrame(list(sales_velocities.items()))
    sales_velocities.columns = ["vpn", "weekly_sales"]
    sales_velocities["sales_vel_pred"] = sales_velocities["weekly_sales"] * len(
        pd.date_range(te_start, te_end, freq="W-MON"))

    # join table
    result = sales_velocities[["vpn", "sales_vel_pred"]].merge(agg_testing_error, how="left", on="vpn")
    result["vel_mape (%)"] = abs(result["sales_vel_pred"] / result["gt"] - 1) * 100
    result = result[["vpn", "gt", "sales_vel_pred", "vel_mape (%)", "model_pred", "model_mape (%)"]]
    result["info"] = cluster

    report.append(result)

report = pd.concat(report)
report.to_csv(os.path.join(result_path, "model_report.csv"), index=False)

# COMMAND ----------
