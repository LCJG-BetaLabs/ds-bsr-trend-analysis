# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("mode", "train")
dbutils.widgets.text("fh", "12")
dbutils.widgets.text("tr_end", "")
dbutils.widgets.text("te_start", "")
dbutils.widgets.text("te_end", "")

mode = getArgument("mode")
fh = int(getArgument("fh"))

# COMMAND ----------

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from bsr_trend.utils.data import get_sales_table, get_time_series
from bsr_trend.utils.catalog import init_directory

sales = get_sales_table()
result_path = init_directory(mode=mode, model_name="sales_vel")
start, end = sales["order_week"].min(), sales["order_week"].max()
tr_start, tr_end = start.strftime("%Y-%m-%d"), getArgument("tr_end")
te_start, te_end = getArgument("te_start"), getArgument("te_end")

# COMMAND ----------

vpns = np.unique(sales["vpn"])

if mode == "train":
    # get ground truth
    gt = []
    for vpn in tqdm(vpns):
        subdf = sales[sales["vpn"] == vpn]
        _, tes = get_time_series(subdf, dynamic_start=False, start_date=te_start, end_date=te_end)
        gt.append([vpn, sum(tes[0])])

    gt = pd.DataFrame(gt, columns=["vpn", "gt"])

# COMMAND ----------

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
sales_velocities["sales_vel_pred"] = sales_velocities["weekly_sales"] * fh

# COMMAND ----------

if mode == "train":
    # join table
    result = gt.merge(sales_velocities[["vpn", "sales_vel_pred"]], how="left", on="vpn")
    result["vel_mape (%)"] = abs(result["sales_vel_pred"] / result["gt"] - 1) * 100
    result = result[["vpn", "gt", "sales_vel_pred", "vel_mape (%)"]]
    # save
    result.to_csv(os.path.join(result_path, "model_report.csv"), index=False)
elif mode == "predict":
    result = sales_velocities[["vpn", "sales_vel_pred"]]
    result.to_csv(os.path.join(result_path, "model_report.csv"), index=False)
