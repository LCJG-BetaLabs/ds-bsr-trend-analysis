# Databricks notebook source
import os
import warnings
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

from bsr_trend.models.ets_model import ETSModel
from bsr_trend.models.croston_model import CrostonModel
from bsr_trend.utils.data import get_sales_table
from bsr_trend.logger import get_logger
from bsr_trend.utils.catalog import CUTOFF_DATE

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")

logger = get_logger()

# COMMAND ----------

dbutils.widgets.removeAll()
# format: yyyy-MM-dd
# default: today
dbutils.widgets.text("cutoff_date", CUTOFF_DATE) 

# COMMAND ----------

cutoff_date = getArgument("cutoff_date")

# COMMAND ----------

sales = get_sales_table()

# train test split
start, end = sales["order_week"].min(), sales["order_week"].max()
tr_start, tr_end = start.strftime("%Y-%m-%d"), (datetime.datetime.strptime(cutoff_date, "%Y-%m-%d").date() - relativedelta(months=3)).strftime("%Y-%m-%d")
te_start, te_end = tr_end, cutoff_date

logger.info(f"""num of vpn: {len(sales["vpn"].unique())}""")

# COMMAND ----------

# validate
for vpn in np.unique(sales["vpn"]):
    subdf = sales[sales["vpn"] == vpn]
    # at least 3 months of training data
    if datetime.datetime.date(subdf["order_week"].min()) > datetime.datetime.strptime(tr_end, "%Y-%m-%d").date() - relativedelta(months=3):
        sales = sales[~(sales["vpn"] == vpn)]
    # at least 3 record of data
    if len(subdf) <= 3:
        sales = sales[~(sales["vpn"] == vpn)]

# COMMAND ----------

# train
ets = ETSModel(
    data=sales,
    tr_start=tr_start,
    tr_end=tr_end, 
    te_start=te_start,
    te_end=te_end, 
    mode="train",
    model_name="ets_models"
)
ets.train_predict_evaluate()

# COMMAND ----------

croston = CrostonModel(
    data=sales,
    tr_start=tr_start,
    tr_end=tr_end, 
    te_start=te_start,
    te_end=te_end, 
    mode="train",
    model_name="croston_models"
)
croston.train_predict_evaluate()

# COMMAND ----------

# sales vel
dbutils.notebook.run(
    "./bsr_trend/models/sales_vel", 
    0, 
    {
        "mode": "train",
        "fh": 12,
        "tr_end": tr_end,
        "te_start": te_start,
        "te_end": te_end,
    }
)

# COMMAND ----------

# combine and save best model report
# TODO: save_combined_model_report(model_list=["sales_vel", "croston_models", "ets_models"])
dbutils.notebook.run(
    "./bsr_trend/models/evaluate", 
    0,
)

# COMMAND ----------


