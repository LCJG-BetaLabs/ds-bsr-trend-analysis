# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("cluster_no", "")
dbutils.widgets.text("clustering_method", "")

cluster_no = getArgument("cluster_no")
clustering_method = getArgument("clustering_method")

# COMMAND ----------

import base64
import json
import os
import warnings
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

from bsr_trend.model_utils import choose_best_hyperparameter
from bsr_trend.logger import get_logger
from bsr_trend.exog_data import get_weekly_traffic, tag_holidays, sales_period

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")

logger = get_logger()
weekly_traffic = get_weekly_traffic().toPandas()

# COMMAND ----------

# for cluster 0 - cluster 0
path = f"/dbfs/mnt/dev/bsr_trend/clustering/{clustering_method}/"
cluster_result = pd.read_csv(os.path.join(path, f"cluster_mapping.csv"))

result_path = os.path.join(path, "sarimax_result", f"cluster_{cluster_no}")
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

# load data
sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
sales["order_week"] = pd.to_datetime(sales["order_week"])

sales = cluster_result[["vpn"]].merge(sales, how="left", on="vpn")
start, end = sales["order_week"].min(), sales["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# train_test_split
tr_start, tr_end = start.strftime('%Y-%m-%d'), '2023-09-01'
te_start, te_end = '2023-09-01', '2023-11-30'
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

# train_test_split
pred_date_range = pd.date_range(pred_start, pred_end, freq="W-MON")
pred_buf = pd.DataFrame(index=pred_date_range).reset_index()
pred_buf.columns = ["order_week"]
pred_buf = pred_buf.set_index("order_week")

# COMMAND ----------

# check if the training period >= 52*2 weeks
# for the SARIMAX model to work
_tr_start = datetime.datetime.strptime(tr_start, "%Y-%m-%d")
_tr_end = datetime.datetime.strptime(tr_end, "%Y-%m-%d")
diff = (_tr_end - _tr_start).days // 7

if diff < 104:
    dbutils.notebook.exit(f"Only {diff} weeks in the training data")

# COMMAND ----------

vpns = np.unique(sales["vpn"])
# fill 0 so that the length of each series is the same
tras = []
tess = []
for vpn in vpns:
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
    buf = pd.merge(
        pd.DataFrame(index=date_range),
        subdf,
        how="left",
        left_index=True,
        right_index=True,
    )
    buf["order_week"] = buf.index
    buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
    buf = buf.fillna(0).astype(int)

    # take training period
    tra = buf['qty'][tr_start:tr_end].dropna()
    tes = buf['qty'][te_start:te_end].dropna()
    tras.append(tra)
    tess.append(tes)

# COMMAND ----------

# remove items that has 0 sales in testing period
zero_ts_index = [i for i, t in enumerate(tess) if sum(t) == 0]
tras = [value for i, value in enumerate(tras) if i not in zero_ts_index]
tess = [value for i, value in enumerate(tess) if i not in zero_ts_index]
vpns = np.delete(vpns, zero_ts_index)
print(f"removed {vpns[zero_ts_index]}")

# COMMAND ----------

tra_avg = pd.concat(tras, axis=1).mean(axis=1)
tes_avg = pd.concat(tess, axis=1).mean(axis=1)

# COMMAND ----------

tra_avg.plot()

# COMMAND ----------

tes_avg.plot()

# COMMAND ----------

best_p, best_d, best_q, best_P, best_D, best_Q = choose_best_hyperparameter(tra_avg)

# COMMAND ----------

def prepare_training_data(vpn, pred_buf):
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
    buf = pd.merge(
        pd.DataFrame(index=date_range),
        subdf,
        how="left",
        left_index=True,
        right_index=True,
    )
    buf["order_week"] = buf.index

    # dynamic start date for each product
    tr_start = buf[buf.vpn.notna()]["order_week"].min().to_pydatetime()
    logger.info(f"first purchase date: {tr_start}")
    if tr_start > datetime.datetime.strptime(tr_end, '%Y-%m-%d') - datetime.timedelta(days=3 * 30):
        # at least 3 months of training data, skip 
        logger.info(f"item '{vpn}' is skipped")
        status = "skip"
        return None, None, None, None, None, status
    elif tr_start < datetime.datetime(2021, 9, 1):
        tr_start = "2021-09-01"
        status = ">=2yr"
    else:
        tr_start = tr_start.strftime('%Y-%m-%d')
    logger.info(f"train start: {tr_start}")

    buf = buf.drop(["vpn", "amt", "order_week"], axis=1)
    buf = buf.fillna(0).astype(int)

    # train test split
    tra = buf['qty'][tr_start:tr_end].dropna()
    tes = buf['qty'][te_start:te_end].dropna()

    # exog variable
    # weekly_traffic
    wt = weekly_traffic.set_index("week_start_date")["weekly_traffic"]

    # holidays
    buf["order_week"] = buf.index
    buf['order_week'] = pd.to_datetime(buf['order_week'])
    buf['holiday_count'] = buf.groupby(pd.Grouper(key='order_week', freq='W-MON'))['order_week'].transform(
        lambda x: sum(x.map(tag_holidays)))

    # sales
    buf["is_sales_period"] = buf["order_week"].apply(lambda d: sales_period(d))

    buf["weekly_traffic"] = wt

    exog = buf[["holiday_count", "is_sales_period", "weekly_traffic"]]
    exog_train = exog[tr_start:tr_end].dropna()
    exog_test = exog[te_start:te_end].dropna()
    exog_pred = pred_buf.set_index("order_week")[["holiday_count", "is_sales_period", "weekly_traffic"]].dropna()

    return tra, tes, exog_train, exog_test, exog_pred


# COMMAND ----------

# train
for vpn in tqdm(vpns, total=len(vpns)):
    tra, tes, exog_train, exog_test, exog_pred = prepare_training_data(vpn, pred_buf)
    # train model
    model = sm.tsa.statespace.SARIMAX(
        tra,
        order=(best_p, best_d, best_q),
        seasonal_order=(best_P, best_D, best_Q, 52),
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    test = model.get_prediction(min(tes.index), max(tes.index), exog=exog_test)
    # test = model.get_prediction(te_start, te_end, exog=exog_test)
    test_pred = test.predicted_mean.fillna(0)

    # metrics
    tescopy = tes.copy()
    tescopy[tescopy == 0] = 0.1
    mse = mean_squared_error(tescopy, test_pred)
    mape = mean_absolute_percentage_error(tescopy, test_pred)

    # forecast
    prediction = model.predict(min(pred_buf.index), max(pred_buf.index), exog=exog_pred)
    # prediction = model.predict(pred_start, pred_end, exog=exog_pred)

    # save results
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = os.path.join(result_path, encoded_vpn)
    os.makedirs(folder, exist_ok=True)

    tra.to_csv(f"{folder}/dataset_train.csv")
    tes.to_csv(f"{folder}/dataset_test.csv")
    test_pred.to_csv(f"{folder}/dataset_prediction_on_test.csv")
    prediction.to_csv(f"{folder}/prediction.csv")
    exog_train.to_csv(f"{folder}/exog_train.csv")
    exog_test.to_csv(f"{folder}/exog_test.csv")
    exog_pred.to_csv(f"{folder}/exog_pred.csv")

    with open(f"{folder}/metrics.txt", "w") as f:
        f.write(json.dumps({"mse": mse, "mape": mape}, indent=None))
    sm.iolib.smpickle.save_pickle(model, f"{folder}/arima.pkl")
    with open(f"{folder}/statsmodels.version", "w") as f:
        f.write(sm.__version__)

# COMMAND ----------

# agg test pred and evaluate
# mape for testing period, for each product
test_and_preds = []
all_tests = []
all_test_preds = []
mapes = []
for vpn in tqdm(vpns, total=len(vpns)):
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = os.path.join(result_path, encoded_vpn)
    tes = pd.read_csv(f"{folder}/dataset_test.csv")
    test = pd.read_csv(f"{folder}/dataset_prediction_on_test.csv")
    error = mean_absolute_percentage_error(tes["qty"], test["predicted_mean"])
    test_and_pred = [vpn, sum(tes["qty"]), sum(test["predicted_mean"])]
    test_and_preds.append(test_and_pred)
    mapes.append(error)
    all_tests.append(tes)
    all_test_preds.append(test)

# COMMAND ----------

agg_testing_error = pd.DataFrame(test_and_preds, columns=["vpn", "gt", "model_pred"])
agg_testing_error["model_mape (%)"] = abs(agg_testing_error["model_pred"] / (agg_testing_error["gt"] + 0.01) - 1) * 100
display(agg_testing_error)

# COMMAND ----------

mape = mean_absolute_percentage_error(agg_testing_error["test_qty"], agg_testing_error["test_pred"])
mape_df = pd.DataFrame([mape], columns=["mape"])

# COMMAND ----------

display(mape_df)

# COMMAND ----------

testing_error = pd.DataFrame(zip(vpns, mapes), columns=["vpn", "mapes"])

# COMMAND ----------

display(testing_error)

# COMMAND ----------

# sales velocities

sales_velocities = {}
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
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
    # sales velocity for 
    buf = buf[(buf["order_week"] >= tr_start) & (buf["order_week"] <= tr_end)]
    sales_velocity = buf["qty"].mean()
    sales_velocities[vpn] = sales_velocity

sales_velocities = pd.DataFrame(list(sales_velocities.items()))
sales_velocities.columns = ["vpn", "weekly_sales"]
sales_velocities["sales_vel_pred"] = sales_velocities["weekly_sales"] * len(pd.date_range(te_start, te_end, freq="W-MON"))

# COMMAND ----------

pred_sales_velocities = {}
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
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
vel_pred["vel_mape (%)"] = abs(vel_pred["sales_vel_pred"] / (vel_pred["gt"] + 0.01) - 1) * 100

# COMMAND ----------

display(vel_pred)

# COMMAND ----------

# join agg_testing_error and vel_pred of testing period
result = vel_pred[["vpn", "sales_vel_pred", "vel_mape (%)"]].merge(agg_testing_error, how="left", on="vpn")
result.to_csv(os.path.join(result_path, "agg_result.csv"), index=False)
