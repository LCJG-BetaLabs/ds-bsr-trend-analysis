# Databricks notebook source
import pandas as pd
import numpy as np
from bsr_trend.utils.data import get_time_series
from bsr_trend.utils.catalog import TREND_CLASSIFICATION_RESULT, write_uc_table
from bsr_trend.trend_classification.func import classify_trend


sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
tr_end = "2023-09-01"
tra = get_time_series(sales, dynamic_start=True, start_date=None, end_date=tr_end)
result = classify_trend(tra)
np.unique(result)

# COMMAND ----------

# save result
vpns = np.unique(sales["vpn"])
result = pd.DataFrame({"vpn": vpns, "trend_class": result, "train_end": tr_end})

write_uc_table(
    TREND_CLASSIFICATION_RESULT,
    result,
    mode="overwrite",
)