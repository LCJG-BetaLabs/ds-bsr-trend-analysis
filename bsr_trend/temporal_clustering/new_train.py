# Databricks notebook source
import base64
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
from tqdm import tqdm

from bsr_trend.model_utils import choose_best_hyperparameter
from bsr_trend.logger import get_logger
from bsr_trend.exog_data import get_weekly_traffic, tag_holidays, sales_period
from bsr_trend.utils.data import get_sales_table, get_time_series
from bsr_trend.utils.catalog import CLUSTERING_MAPPING

# Suppress UserWarning from statsmodels
warnings.simplefilter("ignore")
logger = get_logger()

path = f"/dbfs/mnt/dev/bsr_trend/clustering/kmeans_dtw/"
result_path = os.path.join(path, "sarimax_result")
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

# load data
sales = get_sales_table()
cluster_mapping = spark.table(CLUSTERING_MAPPING).toPandas()
sales = sales.merge(cluster_mapping[["vpn", "cluster"]], on="vpn", how="left")
sales = sales[~sales["cluster"].isna()]

# COMMAND ----------

start, end = sales["order_week"].min(), sales["order_week"].max()

# train_test_split
tr_start, tr_end = start.strftime("%Y-%m-%d"), "2023-09-01"
te_start, te_end = "2023-09-01", "2023-11-30"
pred_start, pred_end = te_start, "2024-02-29"
real_pred_start = "2023-12-01"

# COMMAND ----------

weekly_traffic = get_weekly_traffic().toPandas()

# COMMAND ----------


def select_hyperparameter(tra, save_path):
    hp_path = os.path.join(save_path, "hyperparameters.json")

    if os.path.exists(hp_path):
        with open(hp_path) as file:
            data = json.load(file)
        hyperparameters = {
            "best_p": data['best_p'],
            "best_d": data['best_d'],
            "best_q": data['best_q'],
            "best_P": data['best_P'],
            "best_D": data['best_D'],
            "best_Q": data['best_Q'],
        }
    else:
        best_p, best_d, best_q, best_P, best_D, best_Q = choose_best_hyperparameter(tra)
        hyperparameters = {
            "best_p": best_p,
            "best_d": best_d,
            "best_q": best_q,
            "best_P": best_P,
            "best_D": best_D,
            "best_Q": best_Q,
        }
        # save best hyperparameter
        with open(hp_path, "w") as file:
            json.dump(hyperparameters, file)
    return hyperparameters


def get_exog_variable(start_date, end_date):
    date_range = pd.date_range(start_date, end_date, freq="W-MON")
    buf = pd.DataFrame(index=date_range)
    buf["order_week"] = buf.index

    # weekly_traffic
    wt = weekly_traffic.set_index("week_start_date")["weekly_traffic"]

    # holidays
    buf["order_week"] = pd.to_datetime(buf["order_week"])
    buf["holiday_count"] = buf.groupby(pd.Grouper(key="order_week", freq="W-MON"))["order_week"].transform(
        lambda x: sum(x.map(tag_holidays)))

    # sales period
    buf["is_sales_period"] = buf["order_week"].apply(lambda d: sales_period(d))

    buf = buf.join(wt, how="left")
    exog = buf[["holiday_count", "is_sales_period", "weekly_traffic"]]
    exog = exog[start_date:end_date].dropna()
    return exog


def train(vpn, tra, tes, exog_train, exog_test, exog_pred, hyperparameters, save_path):
    model = sm.tsa.statespace.SARIMAX(
        tra,
        order=(hyperparameters["best_p"], hyperparameters["best_d"], hyperparameters["best_q"]),
        seasonal_order=(hyperparameters["best_P"], hyperparameters["best_D"], hyperparameters["best_Q"], 52),
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    test = model.get_prediction(min(tes.index), max(tes.index), exog=exog_test)
    test_pred = test.predicted_mean.fillna(0)

    # metrics
    tescopy = tes.copy()
    tescopy[tescopy == 0] = 0.1
    mse = mean_squared_error(tescopy, test_pred)
    mape = mean_absolute_percentage_error(tescopy, test_pred)

    # forecast
    prediction = model.predict(min(exog_pred.index), max(exog_pred.index), exog=exog_pred)

    # save results
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = os.path.join(save_path, encoded_vpn)
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

    logger.info(f"Results saved to {folder}")


def evaluate():
    ...
    return


# COMMAND ----------

distinct_cluster = np.unique(sales["cluster"])

for cluster in tqdm(distinct_cluster):
    subdf = sales[sales["cluster"] == cluster]
    tra = get_time_series(subdf, dynamic_start=False, start_date=None, end_date=tr_end)
    tes = get_time_series(subdf, dynamic_start=False, start_date=te_start, end_date=te_end)

    # get cluster avg for selecting hyperparameter
    tra_avg = pd.concat(tra, axis=1).mean(axis=1)
    tes_avg = pd.concat(tes, axis=1).mean(axis=1)

    # select hyperparameter and save
    save_path = os.path.join(result_path, cluster)
    os.makedirs(save_path, exist_ok=True)
    hyperparameters = select_hyperparameter(tra_avg, save_path=save_path)

    # exog variable
    exog_train = get_exog_variable(start_date=tr_start, end_date=tr_end)
    exog_test = get_exog_variable(start_date=te_start, end_date=te_end)
    exog_pred = get_exog_variable(start_date=pred_start, end_date=pred_end)

    # train
    vpns = np.unique(subdf["vpn"])
    for vpn, _tra, _tes in zip(vpns, tra, tes):
        train(vpn, _tra, _tes, exog_train, exog_test, exog_pred, hyperparameters, save_path)

    # evaluate (excel report as output)
    evaluate()

# COMMAND ----------


