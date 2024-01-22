# Databricks notebook source
# MAGIC %md
# MAGIC # load data

# COMMAND ----------

import base64
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm
import statsmodels.api as sm

from bsr_trend.model_utils import choose_best_hyperparameter
from bsr_trend.logger import get_logger
from bsr_trend.exog_data import one_hot_encode_month

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")

logger = get_logger()

# COMMAND ----------

df = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
df["order_week"] = pd.to_datetime(df["order_week"])
df["vpn"].nunique()

# COMMAND ----------

start, end = df["order_week"].min(), df["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# COMMAND ----------

# train_test_split
tr_start, tr_end = '2019-12-30', '2023-05-01'
te_start, te_end = '2023-05-08', '2023-11-06'
pred_start, pred_end = te_start, "2024-05-06"
real_pred_start = "2023-11-13"

# COMMAND ----------

# MAGIC %md
# MAGIC model for each cluster

# COMMAND ----------

# train_test_split
pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]

pred_buf = one_hot_encode_month(pred_buf, "order_week")
pred_buf = pred_buf.set_index("order_week")

exog_pred = pred_buf

# COMMAND ----------

path = "/dbfs/mnt/dev/bsr_trend/clustering/clustering_result/"
cluster_mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))

# COMMAND ----------

cluster_num = 2

# COMMAND ----------


def random_ts_from_cluster(cluster_num):
    vpns = cluster_mapping[cluster_mapping["cluster"] == cluster_num]["vpn"].values
    return np.random.choice(vpns, 1)[0]


# COMMAND ----------

len(cluster_mapping[cluster_mapping["cluster"] == cluster_num])

# COMMAND ----------

# get a random ts from the cluster
random_vpn = random_ts_from_cluster(cluster_num)

subdf = df[df["vpn"] == random_vpn].set_index("order_week")
buf = pd.merge(
    pd.DataFrame(index=date_range),
    subdf,
    how="left",
    left_index=True,
    right_index=True,
)
buf["order_week"] = buf.index
buf = one_hot_encode_month(buf, "order_week")
buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
buf = buf.fillna(0).astype(int)

# train test split
tra = buf['qty'][tr_start:tr_end].dropna()
tes = buf['qty'][te_start:te_end].dropna()
exog_train = buf.drop(["qty"], axis=1)[tr_start:tr_end].dropna()
exog_test = buf.drop(["qty"], axis=1)[te_start:te_end].dropna()


# COMMAND ----------

# choose best hyperparameter
best_p, best_d, best_q, best_P, best_D, best_Q = choose_best_hyperparameter(tra)

# COMMAND ----------

temp = cluster_mapping[cluster_mapping["cluster"] == cluster_num]
temp

# COMMAND ----------

# train the model using the selected parameter

# test run 1 cluster
temp = cluster_mapping[cluster_mapping["cluster"] == cluster_num]
vpns = np.unique(temp["vpn"])

for vpn in tqdm(vpns, total=len(vpns)):
    subdf = df[df["vpn"] == vpn].set_index("order_week")
    buf = pd.merge(
        pd.DataFrame(index=date_range),
        subdf,
        how="left",
        left_index=True,
        right_index=True,
    )
    buf["order_week"] = buf.index
    buf = one_hot_encode_month(buf, "order_week")
    buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
    buf = buf.fillna(0).astype(int)

    # train test split
    tra = buf['qty'][tr_start:tr_end].dropna()
    tes = buf['qty'][te_start:te_end].dropna()
    exog_train = buf.drop(["qty"], axis=1)[tr_start:tr_end].dropna()
    exog_test = buf.drop(["qty"], axis=1)[te_start:te_end].dropna()

    # train model
    model = sm.tsa.statespace.SARIMAX(
        tra,
        # order=(4,1,5),
        # seasonal_order=(4,1,5,52),
        order=(best_p, best_d, best_q),
        seasonal_order=(best_P, best_D, best_Q, 52),
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    test = model.get_prediction(te_start, te_end, exog=exog_test)
    test_pred = test.predicted_mean.fillna(0)

    # metrics
    tescopy = tes.copy()
    tescopy[tescopy == 0] = 0.1
    mse = mean_squared_error(tescopy, test_pred)
    mape = mean_absolute_percentage_error(tescopy, test_pred)

    # forecast
    prediction = model.predict(pred_start, pred_end, exog=exog_pred)

    # save results
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = f"/dbfs/mnt/dev/bsr_trend/clustering/cluster_{cluster_num}_arima_forecasting/{encoded_vpn}"
    os.makedirs(folder, exist_ok=True)
    buf.to_csv(f"{folder}/dataset.csv")
    tra.to_csv(f"{folder}/dataset_train.csv")
    tes.to_csv(f"{folder}/dataset_test.csv")
    prediction.to_csv(f"{folder}/prediction.csv")
    with open(f"{folder}/metrics.txt", "w") as f:
        f.write(json.dumps({"mse": mse, "mape": mape}, indent=None))
    sm.iolib.smpickle.save_pickle(model, f"{folder}/arima.pkl")
    with open(f"{folder}/statsmodels.version", "w") as f:
        f.write(sm.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluation

# COMMAND ----------

# sales vel * 6 months
sales_velocities = {}
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = df[df["vpn"] == vpn].set_index("order_week")
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
sales_velocities["forecast"] = sales_velocities["weekly_sales"] * len(pd.date_range(te_start, te_end, freq="W-MON"))
sales_velocities

# COMMAND ----------

pred_sales_velocities = {}
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = df[df["vpn"] == vpn].set_index("order_week")
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
    buf = buf[(buf["order_week"] >= te_start) & (buf["order_week"] <= te_end)]
    sales_velocity = buf["qty"].sum()
    pred_sales_velocities[vpn] = sales_velocity

pred_sales_velocities = pd.DataFrame(list(pred_sales_velocities.items()))
pred_sales_velocities.columns = ["vpn", "gt"]
pred_sales_velocities

# COMMAND ----------

vel_pred = pd.merge(sales_velocities, pred_sales_velocities, how="inner", on="vpn")
vel_pred = vel_pred.drop(columns="weekly_sales")
vel_pred["mape (%)"] = abs(vel_pred["forecast"] / (vel_pred["gt"] + 0.01) - 1) * 100
vel_pred

# COMMAND ----------

# evaluate result
import base64
import glob
import json
import os

import numpy as np
import pandas as pd

preds = []

for path in glob.glob(f"/dbfs/mnt/dev/bsr_trend/clustering/cluster_{cluster_num}_arima_forecasting/*"):
    encoded = os.path.basename(path)
    vpn = base64.b64decode(encoded).decode()

    pred = pd.read_csv(f"{path}/prediction.csv")
    pred.columns = ["week", vpn]
    pred = pred.set_index("week").T
    preds.append(pred)

preds = pd.concat(preds)
preds = preds.clip(lower=0).astype(float)

preds

# COMMAND ----------

metrics = []

for path in glob.glob(f"/dbfs/mnt/dev/bsr_trend/clustering/cluster_{cluster_num}_arima_forecasting/*"):
    encoded = os.path.basename(path)
    vpn = base64.b64decode(encoded).decode()

    with open(f"{path}/metrics.txt", "r") as f:
        m = json.load(f)
    mape = m["mape"]
    metrics.append({
        "vpn": vpn,
        "mape (%)": np.round(mape * 100, 2),
    })

metrics = pd.DataFrame(metrics)

# COMMAND ----------

forecast = preds

# COMMAND ----------

coverage = pd.read_csv("/dbfs/mnt/dev/bsr_trend/week_coverage.csv")
forecast = forecast[[c for c in forecast.columns if c >= "2023-11-13"]]  # what is this date
forecast = forecast.apply(sum, axis=1).reset_index()
forecast.columns = ["vpn", "forecast"]

vpn_min_qty = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_min_qty.csv")  # waht is this

# COMMAND ----------

vel_pred = vel_pred[["vpn", "forecast", "mape (%)"]].rename(
    columns={"mape (%)": "vel_mape (%)", "forecast": "vel_forecast"})

# COMMAND ----------

df = pd.merge(coverage, forecast, how="inner", on="vpn")
df = pd.merge(df, metrics, how="inner", on="vpn")
df = pd.merge(df, vpn_min_qty, how="left", on="vpn")
df = pd.merge(df, vel_pred, how="left", on="vpn")

df["min_qty_hk"] = df["min_qty_hk"].fillna(0).astype(int)
df["forecast"] = np.ceil(df["forecast"]).astype(int)
df["vel_forecast"] = np.ceil(df["vel_forecast"]).astype(int)
df["stock_level"] = df["stock_level"].astype(int)
df["po_qty"] = (df["forecast"] - df["stock_level"] + df["min_qty_hk"]).clip(lower=0)
df["vel_po_qty"] = (df["vel_forecast"] - df["stock_level"] + df["min_qty_hk"]).clip(lower=0)
df = df[
    ["vpn", "supplier", "category", "class", "subclass", "mape (%)", "stock_level", "min_qty_hk", "forecast", "po_qty",
     "vel_forecast", "vel_po_qty", "vel_mape (%)"]]
df

# COMMAND ----------

subclass_mape = df.groupby(["category", "class", "subclass"])["mape (%)", "vel_mape (%)"].mean().reset_index()
subclass_mape

# COMMAND ----------

# save result
subclass_mape.to_csv(f"/dbfs/mnt/dev/bsr_trend/clustering/{cluster_num}_subclass_mape.csv", index=False)

# COMMAND ----------
