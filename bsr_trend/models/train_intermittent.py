# Databricks notebook source
pip install statsforecast

# COMMAND ----------

import os
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from statsforecast import StatsForecast
from statsforecast.models import ADIDA, CrostonClassic, IMAPA, TSB
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

train = get_time_series(sales, dynamic_start=True, start_date=None, end_date=tr_end)
test = get_time_series(sales, dynamic_start=False, start_date=te_start, end_date=te_end)

# COMMAND ----------

vpns = np.unique(sales["vpn"])

# COMMAND ----------

train_list = []
test_list = []
for i, vpn in enumerate(vpns):
    train[i] = pd.DataFrame(train[i], columns=["qty"])
    train[i]["unique_id"] = vpn
    train[i] = train[i].reset_index().rename(columns={"index": "ds"})

    test[i] = pd.DataFrame(test[i], columns=["qty"])
    test[i]["unique_id"] = vpn
    test[i] = test[i].reset_index().rename(columns={"index": "ds"})

# COMMAND ----------

train = pd.concat(train)
test = pd.concat(test)

# COMMAND ----------


model = StatsForecast(
    models=[ADIDA(), CrostonClassic(), IMAPA(), TSB(alpha_d=0.2, alpha_p=0.2)],
    freq="W-MON",
    n_jobs=-1,
)


# COMMAND ----------

train = train.rename(columns={"qty": "y"})

# COMMAND ----------

model.fit(train)

# COMMAND ----------

p = model.predict(h=12)

# COMMAND ----------

p = p.reset_index().merge(test, on=['ds', 'unique_id'], how='left')

# COMMAND ----------

p = p.drop(columns="ds")

# COMMAND ----------

result = p.groupby("unique_id").sum()
result = result.rename(columns={"qty": "gt"})
result["ADIDA_mape (%)"] = abs(result["ADIDA"] / result["gt"] - 1) * 100
result["CrostonClassic_mape (%)"] = abs(result["ADIDA"] / result["gt"] - 1) * 100
result["IMAPA_mape (%)"] = abs(result["ADIDA"] / result["gt"] - 1) * 100
result["TSB_mape (%)"] = abs(result["ADIDA"] / result["gt"] - 1) * 100

# COMMAND ----------

result.to_csv("/dbfs/mnt/dev/bsr_trend/intermittent_model_report.csv")
