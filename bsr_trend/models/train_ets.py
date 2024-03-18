# Databricks notebook source
pip install sktime

# COMMAND ----------

import os
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sktime.forecasting.ets import AutoETS

from bsr_trend.utils.data import get_sales_table, get_time_series


result_path = f"""/dbfs/mnt/dev/bsr_trend/ets_model_{str(datetime.datetime.today().date()).replace("-", "")}/"""
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

sales = get_sales_table()

start, end = sales["order_week"].min(), sales["order_week"].max()
# train_test_split
tr_start, tr_end = start.strftime("%Y-%m-%d"), "2023-09-01"
te_start, te_end = "2023-09-01", "2023-11-30"
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

# COMMAND ----------

pd.options.mode.chained_assignment = None

# COMMAND ----------

vpns = np.unique(sales["vpn"])

gt_and_pred = []
for vpn in tqdm(vpns):
    subdf = sales[sales["vpn"] == vpn]
    tra = get_time_series(subdf, dynamic_start=True, start_date=None, end_date=tr_end)
    tes = get_time_series(subdf, dynamic_start=False, start_date=te_start, end_date=te_end)

    if len(tra[0]) > 24: # more than 6 months
        forecaster = AutoETS(auto=True, n_jobs=-1)
        forecaster.fit(tra[0])
        pred = forecaster.predict(fh=range(1, 12))

        # evaluate
        test_and_pred = [vpn, sum(tes[0]), sum(pred)]
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
# save
result.to_csv(os.path.join(result_path, "model_report.csv"), index=False)

# COMMAND ----------


