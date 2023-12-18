# Databricks notebook source
import base64
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm


# COMMAND ----------

df = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
df["order_week"] = pd.to_datetime(df["order_week"])
df["vpn"].nunique()

# COMMAND ----------

start, end = df["order_week"].min(), df["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# COMMAND ----------

def one_hot_encode_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[date_col].apply(lambda x: x.month)
    month_dummies = pd.get_dummies(df['month'], prefix="month", prefix_sep="-")
    for i in range(1, 13):
        col = f"month-{i}"
        if col not in month_dummies.columns:
            month_dummies[col] = 0
    month_dummies = month_dummies[[f"month-{i}" for i in range(1, 13)]]
    df = df.drop(columns=[c for c in month_dummies.columns if c in df.columns])
    df = pd.concat([df, month_dummies], axis=1).drop(['month'],axis=1)
    return df

# COMMAND ----------

df

# COMMAND ----------

#train_test_split
tr_start, tr_end = '2019-12-30', '2023-05-01'
te_start, te_end = '2023-05-08', '2023-11-06'
pred_start, pred_end = te_start, "2024-05-06"
real_pred_start = "2023-11-13"

# COMMAND ----------

pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]

pred_buf = one_hot_encode_month(pred_buf, "order_week")
pred_buf = pred_buf.set_index("order_week")

exog_pred = pred_buf

# COMMAND ----------

# MAGIC %md
# MAGIC # model choice

# COMMAND ----------

vpns = np.unique(df["vpn"])

# COMMAND ----------

# Suppress UserWarning from statsmodels

import warnings
warnings.simplefilter("ignore")

# COMMAND ----------

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
    exog_train = buf.drop(["qty"], axis = 1)[tr_start:tr_end].dropna()
    exog_test = buf.drop(["qty"], axis = 1)[te_start:te_end].dropna()

    # train model
    model = sm.tsa.statespace.SARIMAX(
        tra,
        order=(4,1,5),
        seasonal_order=(4,1,5,52),
        exog=exog_train,
        enforce_stationarity=False, 
        enforce_invertibility=False,
    ).fit()
    test = model.get_prediction(te_start, te_end, exog=exog_test)
    test_pred = test.predicted_mean.fillna(0)

    # metrics
    tescopy = tes.copy()
    tescopy[tescopy==0] = 0.1
    mse = mean_squared_error(tescopy, test_pred)
    mape = mean_absolute_percentage_error(tescopy, test_pred)

    # forecast
    prediction = model.predict(pred_start, pred_end, exog=exog_pred)

    # save results
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = f"/dbfs/mnt/dev/bsr_trend/arima_forecasting/{encoded_vpn}"
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

# MAGIC %md # sales velocity * 6 months

# COMMAND ----------

vpns = np.unique(df["vpn"])

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


# COMMAND ----------

vpns = np.unique(df["vpn"])

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

# COMMAND ----------

vel_pred = pd.merge(sales_velocities, pred_sales_velocities, how="inner", on="vpn")
vel_pred = vel_pred.drop(columns="weekly_sales")
vel_pred["mape (%)"] = abs(vel_pred["forecast"] / (vel_pred["gt"]+0.01) - 1) * 100
vel_pred

# COMMAND ----------

vel_pred.to_csv("/dbfs/mnt/dev/bsr_trend/vel_pred.csv", index=False)

# COMMAND ----------

# MAGIC %md # aggregate

# COMMAND ----------

import base64
import glob
import json
import os

import numpy as np
import pandas as pd

# COMMAND ----------

preds = []

for path in glob.glob("/dbfs/mnt/dev/bsr_trend/arima_forecasting/*"):
    encoded = os.path.basename(path)
    vpn = base64.b64decode(encoded).decode()

    pred = pd.read_csv(f"{path}/prediction.csv")
    pred.columns = ["week", vpn]
    pred = pred.set_index("week").T
    preds.append(pred)

preds = pd.concat(preds)
preds = preds.clip(lower=0).astype(float)

# COMMAND ----------

preds

# COMMAND ----------

preds.to_csv("/dbfs/mnt/dev/bsr_trend/arima_forecasting_result.csv")

# COMMAND ----------

metrics = []

for path in glob.glob("/dbfs/mnt/dev/bsr_trend/arima_forecasting/*"):
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

metrics

# COMMAND ----------

metrics.to_csv("/dbfs/mnt/dev/bsr_trend/arima_forecasting_metrics.csv")

# COMMAND ----------

# MAGIC %md # aggregate

# COMMAND ----------

coverage = pd.read_csv("/dbfs/mnt/dev/bsr_trend/week_coverage.csv")
forecast = pd.read_csv("/dbfs/mnt/dev/bsr_trend/arima_forecasting_result.csv", index_col=0)
metrics = pd.read_csv("/dbfs/mnt/dev/bsr_trend/arima_forecasting_metrics.csv", index_col=0)
forecast = forecast[[c for c in forecast.columns if c >= "2023-11-13"]]
forecast = forecast.apply(sum, axis=1).reset_index()
forecast.columns = ["vpn", "forecast"]

vpn_min_qty = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_min_qty.csv")

# COMMAND ----------

metrics

# COMMAND ----------

vel_pred = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vel_pred.csv")
vel_pred = vel_pred[["vpn", "forecast", "mape (%)"]].rename(columns={"mape (%)": "vel_mape (%)", "forecast": "vel_forecast"})

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
df = df[["vpn", "supplier", "category", "class", "subclass", "mape (%)", "stock_level", "min_qty_hk", "forecast", "po_qty", "vel_forecast", "vel_po_qty"]]
df

# COMMAND ----------

df.to_csv("/dbfs/mnt/dev/bsr_trend/forecasted_po_qty.csv", index=False)

# COMMAND ----------

subclass_mape = df.groupby(["category", "class", "subclass"])["mape (%)", "vel_mape (%)"].mean().reset_index()
subclass_mape

# COMMAND ----------

subclass_mape.to_csv("/dbfs/mnt/dev/bsr_trend/subclass_mape.csv", index=False)

# COMMAND ----------

