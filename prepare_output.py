# Databricks notebook source
pip install openpyxl

# COMMAND ----------

import os
import numpy as np
import pandas as pd
from bsr_trend.utils.catalog import CUTOFF_DATE, ENVIRONMENT, HISTORY_FLAG_RESULT
from bsr_trend.utils.data import get_sales_table

# COMMAND ----------

prediction_result = pd.read_csv(os.path.join(f"/dbfs/mnt/{ENVIRONMENT}/bsr_trend/", f"bsr_prediction_{CUTOFF_DATE}.csv"))

# COMMAND ----------

# validate if every product is here
sales = get_sales_table()
if not (prediction_result['vpn'].isin(sales["vpn"]).all() and sales["vpn"].isin(prediction_result["vpn"]).all()):
    raise ValueError("Products in input and output are not consistant.")
print(len(sales["vpn"].unique()))

# COMMAND ----------

# get history flag
history = spark.table(HISTORY_FLAG_RESULT).toPandas()
prediction_result = prediction_result.merge(history, on="vpn", how="left")

# COMMAND ----------

# get order leadtime and min display qty * num of store hk
bsr_list = pd.read_excel("/dbfs/mnt/prd/bsr_trend/provided_by_bu/_New BSR List_2023.xlsx", sheet_name="SS24 BSR List")
bsr_list = bsr_list[["Vendor Product Number", "HK Min. Display Qty per store", "HK no. of Stores", "Order Leadtime (Weeks)"]].rename(columns={"Vendor Product Number": "vpn"})
bsr_list = bsr_list[~bsr_list["HK Min. Display Qty per store"].isna()]
bsr_list = bsr_list.replace("Online", 0)
bsr_list["HK Total Min. Display Qty"] = bsr_list.apply(lambda row: int(row["HK Min. Display Qty per store"]) * int(row["HK no. of Stores"]), axis=1)
bsr_list = bsr_list[["vpn", "HK Total Min. Display Qty", "Order Leadtime (Weeks)"]]
prediction_result = prediction_result.merge(bsr_list, on="vpn", how="left")

# COMMAND ----------

# adjust with order leadtime
prediction_result["adjusted_predicted_qty"] = prediction_result.apply(
    lambda row: row["predicted_qty"] / 12 * row["Order Leadtime (Weeks)"]
    if not row["Order Leadtime (Weeks)"] != np.nan
    else row["predicted_qty"],
    axis=1,
)

# COMMAND ----------

prediction_result

# COMMAND ----------


