# Databricks notebook source
# MAGIC %pip install autogluon

# COMMAND ----------

import os
import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from bsr_trend.utils.data import get_sales_table, get_time_series

# COMMAND ----------

sales = get_sales_table()

start, end = sales["order_week"].min(), sales["order_week"].max()
# train_test_split
tr_start, tr_end = start.strftime("%Y-%m-%d"), "2023-09-01"
te_start, te_end = "2023-09-01", "2023-11-30"
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

# COMMAND ----------

sales["order_week"] = pd.to_datetime(sales["order_week"])

# COMMAND ----------

sales = sales[["vpn", "order_week", "qty"]]

# COMMAND ----------

tra = get_time_series(sales, dynamic_start=True, start_date=None, end_date=tr_end)

# COMMAND ----------

vpns = np.unique(sales["vpn"])
train_data = []
for vpn, _tra in zip(vpns, tra):
    tmp = pd.DataFrame(_tra)
    tmp["vpn"] = vpn
    train_data.append(tmp)
train_data = pd.concat(train_data)

# COMMAND ----------

train_data["order_week"] = train_data.index

# COMMAND ----------

train_data = TimeSeriesDataFrame.from_data_frame(
    train_data,
    id_column="vpn",
    timestamp_column="order_week",
)
train_data.head()

# COMMAND ----------

sales

# COMMAND ----------

result_path = "/dbfs/mnt/dev/bsr_trend/autogluson/"
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

train_data

# COMMAND ----------

predictor.freq

# COMMAND ----------

train_data.freq = "W-MON"

# COMMAND ----------

predictor = TimeSeriesPredictor(
    prediction_length=12,
    path="/dbfs/mnt/dev/bsr_trend/autogluson/",
    target="qty",
    eval_metric="MAPE",
    freq="W-MON",
)

predictor.fit(
    TimeSeriesDataFrame(train_data),
    presets="best_quality", # best_quality
    # time_limit=600,
)

# COMMAND ----------

predictor

# COMMAND ----------


