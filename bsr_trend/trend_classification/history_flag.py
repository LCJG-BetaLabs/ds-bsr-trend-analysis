# Databricks notebook source
import pandas as pd
import numpy as np
from bsr_trend.utils.data import get_time_series, get_sales_table
from bsr_trend.utils.catalog import write_uc_table, CUTOFF_DATE, HISTORY_FLAG_RESULT
from bsr_trend.trend_classification.func import classify_by_history


sales = get_sales_table()
vpn, tra = get_time_series(sales, dynamic_start=True, start_date=None, end_date=CUTOFF_DATE)

history_result = classify_by_history(tra)
print(np.unique(history_result, return_counts=True))

# COMMAND ----------

# save result
vpns = np.unique(sales["vpn"])
result = pd.DataFrame(
    {
        "vpn": vpns,
        "history": history_result,
        "train_end": CUTOFF_DATE,
    }
)

# COMMAND ----------

write_uc_table(
    HISTORY_FLAG_RESULT,
    result,
    mode="overwrite",
)

# COMMAND ----------


