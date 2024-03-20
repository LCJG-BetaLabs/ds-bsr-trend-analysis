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
from bsr_trend.utils.catalog import BEST_MODEL_REPORT_PATH, PREDICTION_DIR, ENVIRONMENT, CUTOFF_DATE
from bsr_trend.logger import get_logger

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")

logger = get_logger()

# COMMAND ----------

dbutils.widgets.removeAll()
# format: yyyy-MM-dd
# default: today
dbutils.widgets.text("cutoff_date", CUTOFF_DATE) 
cutoff_date = getArgument("cutoff_date")
os.environ["CUTOFF_DATE"] = cutoff_date

# COMMAND ----------

sales = get_sales_table()

# train test split
start, end = sales["order_week"].min(), sales["order_week"].max()
tr_start, tr_end = start.strftime("%Y-%m-%d"), (datetime.datetime.strptime(cutoff_date, "%Y-%m-%d").date() - relativedelta(months=3)).strftime("%Y-%m-%d")
te_start, te_end = tr_end, cutoff_date
pred_start, pred_end = te_start, (datetime.datetime.strptime(cutoff_date, "%Y-%m-%d").date() + relativedelta(months=3)).strftime("%Y-%m-%d")

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

# split by best model report
best_model = pd.read_csv(BEST_MODEL_REPORT_PATH)
ets_vpns = best_model[best_model["best_model"] == "ets"]["vpn"].values
croston_vpns = best_model[best_model["best_model"] == "croston"]["vpn"].values

# COMMAND ----------

# train
ets = ETSModel(
    data=sales[sales["vpn"].isin(ets_vpns)],
    tr_start=tr_start,
    tr_end=tr_end, 
    te_start=te_start,
    te_end=te_end,
    fh=12, 
    mode="predict",
    model_name="ets_models"
)
ets.train_predict_evaluate()

# COMMAND ----------

croston = CrostonModel(
    data=sales[sales["vpn"].isin(croston_vpns)],
    tr_start=tr_start,
    tr_end=tr_end, 
    te_start=te_start,
    te_end=te_end, 
    fh=12, 
    mode="predict",
    model_name="croston_models"
)
croston.train_predict_evaluate()

# COMMAND ----------

# sales vel
dbutils.notebook.run(
    "./bsr_trend/models/sales_vel", 
    0, 
    {
        "mode": "predict",
        "fh": 12,
        "tr_end": te_end,
    }
)

# COMMAND ----------

# final prediction result -> csv in blob
# filename: bsr_prediction_{cutoff_date}.csv
# schema
#  - vpn
#  - round(prediction)

# COMMAND ----------

vel_pred = pd.read_csv(os.path.join(PREDICTION_DIR, "sales_vel", "predictions.csv"))
ets_pred = pd.read_csv(os.path.join(PREDICTION_DIR, "ets_models", "predictions.csv"))
croston_pred = pd.read_csv(os.path.join(PREDICTION_DIR, "croston_models", "predictions.csv"))

result = pd.concat([croston_pred, ets_pred])
vel_pred = vel_pred[~vel_pred["vpn"].isin(np.unique(result["vpn"]))].rename(columns={"sales_vel_pred": "predicted_qty"})
result = pd.concat([result, vel_pred])

result["predicted_qty"] = result["predicted_qty"].round(0)
# handle negative prediction
result["predicted_qty"] = result["predicted_qty"].apply(lambda q: q if q >= 0 else 0)
result.to_csv(os.path.join(PREDICTION_DIR, f"bsr_prediction_{cutoff_date}.csv"), index=False)

os.makedirs(f"/dbfs/mnt/{ENVIRONMENT}/bsr_trend/", exist_ok=True)
result.to_csv(os.path.join(f"/dbfs/mnt/{ENVIRONMENT}/bsr_trend/", f"bsr_prediction_{cutoff_date}.csv"), index=False)
