# Databricks notebook source
# MAGIC %md
# MAGIC evaluate the result for 1 model per vpn 
# MAGIC
# MAGIC random 3 vpn for testing, compare with sales velocities

# COMMAND ----------

import base64
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm
import statsmodels.api as sm

from bsr_trend.model_utils import choose_d, choose_p_and_q, choose_seasonal_p_and_q, extract_season_component
from bsr_trend.logger import get_logger

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")

logger = get_logger()

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
    df = pd.concat([df, month_dummies], axis=1).drop(['month'], axis=1)
    return df
    

# COMMAND ----------

# read data
df = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
df["order_week"] = pd.to_datetime(df["order_week"])
df["vpn"].nunique()

start, end = df["order_week"].min(), df["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# train_test_split
tr_start, tr_end = '2019-12-30', '2023-05-01'
te_start, te_end = '2023-05-08', '2023-11-06'
pred_start, pred_end = te_start, "2024-05-06"
real_pred_start = "2023-11-13"

# train_test_split
pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]

pred_buf = one_hot_encode_month(pred_buf, "order_week")
pred_buf = pred_buf.set_index("order_week")

exog_pred = pred_buf

# COMMAND ----------

def prepare_training_data(vpn):
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

    return tra, tes, exog_train, exog_test

# COMMAND ----------

def choose_best_hyperparameter(tra):
    # choose best hyperparameter
    best_d = choose_d(tra)
    best_D = choose_d(tra, seasonal=True, period=52)
    best_p, best_q = choose_p_and_q(tra, best_d)

    season_component = extract_season_component(tra, period=52)
    # Assuming a seasonal pattern repeating every 52 weeks (annual seasonality)
    # best_P, best_Q = choose_seasonal_p_and_q(tra, best_D, s=52, order=(best_p, best_d, best_q))
    best_P, best_Q = 1, 1
    logger.info(f"Differencing parameter: d = {best_d}")
    logger.info(f"Seasonal differencing parameter: D = {best_D}")
    logger.info(f"AR order and MA order: p = {best_p}, q = {best_q}")
    logger.info(f"Seasonal AR order and MA order: p = {best_P}, q = {best_Q}")

    return best_p, best_d, best_q, best_P, best_D, best_Q

# COMMAND ----------


def train(vpn, best_p, best_d, best_q, best_P, best_D, best_Q):
    # train the model using the selected parameter
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
    folder = f"/dbfs/mnt/dev/bsr_trend/sarimax_forecasting/{encoded_vpn}"
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



# COMMAND ----------



# COMMAND ----------

def get_sales_velocities(vpn):
    # sales vel * 6 months
    sales_velocities = {}
    # for vpn in tqdm(vpns, total=len(vpns)):
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

    pred_sales_velocities = {}
    # for vpn in tqdm(vpns, total=len(vpns)):
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
    
    vel_pred = pd.merge(sales_velocities, pred_sales_velocities, how="inner", on="vpn")
    vel_pred = vel_pred.drop(columns="weekly_sales")
    vel_pred["mape (%)"] = abs(vel_pred["forecast"] / (vel_pred["gt"] + 0.01) - 1) * 100
    return vel_pred



# COMMAND ----------

vel_pred = get_sales_velocities(random_vpns[0])
vel_pred

# COMMAND ----------

def get_mape(vpn):
    metrics = []
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    path = f"/dbfs/mnt/dev/bsr_trend/sarimax_forecasting/{encoded_vpn}"

    with open(f"{path}/metrics.txt", "r") as f:
        m = json.load(f)
    mape = m["mape"]
    metrics.append({
        "vpn": vpn,
        "mape (%)": np.round(mape * 100, 2),
    })
    metrics = pd.DataFrame(metrics)
    return metrics

# COMMAND ----------

get_mape(random_vpns[0])

# COMMAND ----------

random_vpns = df["vpn"].sample(3).values
random_vpns

# COMMAND ----------

for vpn in random_vpns:
    print(vpn)
    tra, tes, exog_train, exog_test = prepare_training_data(vpn)
    best_p, best_d, best_q, best_P, best_D, best_Q = choose_best_hyperparameter(tra)

    train(vpn, best_p, best_d, best_q, best_P, best_D, best_Q)
    vel_pred = get_sales_velocities(vpn)
    print(vel_pred)
    print(get_mape(vpn))

# COMMAND ----------


