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
result_path = os.path.join(path, "arima_result")
os.makedirs(result_path, exist_ok=True)

# COMMAND ----------

from pmdarima.arima import auto_arima

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

# weekly_traffic = get_weekly_traffic().toPandas()

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


def evaluate(vpns, save_path, info=None):
    # agg test pred and evaluate
    # MAPE for testing period, for each product
    gt_and_pred = []
    for vpn in tqdm(vpns, total=len(vpns)):
        encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
        folder = os.path.join(save_path, encoded_vpn)
        tes = pd.read_csv(f"{folder}/dataset_test.csv")
        test = pd.read_csv(f"{folder}/dataset_prediction_on_test.csv")
        test_and_pred = [vpn, sum(tes["qty"]), sum(test["predicted_mean"])]
        gt_and_pred.append(test_and_pred)

    agg_testing_error = pd.DataFrame(gt_and_pred, columns=["vpn", "gt", "model_pred"])
    agg_testing_error["model_mape (%)"] = abs(agg_testing_error["model_pred"] / (agg_testing_error["gt"]) - 1) * 100

    # get sales vel
    sales_velocities = {}
    for vpn in tqdm(vpns, total=len(vpns)):
        subdf = sales[sales["vpn"] == vpn].set_index("order_week")
        start, end = subdf.index.min(), subdf.index.max()
        date_range = pd.date_range(start, end, freq="W-MON")
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
    sales_velocities["sales_vel_pred"] = sales_velocities["weekly_sales"] * len(
        pd.date_range(te_start, te_end, freq="W-MON"))

    # join table
    result = sales_velocities[["vpn", "sales_vel_pred"]].merge(agg_testing_error, how="left", on="vpn")
    result["vel_mape (%)"] = abs(result["sales_vel_pred"] / result["gt"] - 1) * 100
    result = result[["vpn", "gt", "sales_vel_pred", "vel_mape (%)", "model_pred", "model_mape (%)"]]
    result["info"] = info
    # save
    result.to_csv(os.path.join(save_path, "model_report.csv"), index=False)
    display(result)


# COMMAND ----------

distinct_cluster = np.unique(sales["cluster"])

for cluster in tqdm(distinct_cluster[:1]):
    subdf = sales[sales["cluster"] == cluster]
    tra = get_time_series(subdf, dynamic_start=True, start_date=None, end_date=tr_end)
    tes = get_time_series(subdf, dynamic_start=False, start_date=te_start, end_date=te_end)

    # get cluster avg for selecting hyperparameter
    # tra_avg = pd.concat(tra, axis=1).mean(axis=1)
    # tes_avg = pd.concat(tes, axis=1).mean(axis=1)

    # save_path = os.path.join(result_path, cluster)
    # os.makedirs(save_path, exist_ok=True)
    vpns = np.unique(subdf["vpn"])
    gt_and_pred = []
    for vpn, _tra, _tes in zip(vpns, tra, tes):
        model = auto_arima(_tra, seasonal=True, m=52, trace=True)
        print(model.summary())
        predictions = model.predict(n_periods=12)
        test_and_pred = [vpn, sum(_tes), sum(predictions)]
        print(test_and_pred)
        gt_and_pred.append(test_and_pred)

    agg_testing_error = pd.DataFrame(gt_and_pred, columns=["vpn", "gt", "model_pred"])
    agg_testing_error["model_mape (%)"] = abs(agg_testing_error["model_pred"] / (agg_testing_error["gt"]) - 1) * 100
                                    
    # if len(tra_avg) >= 104:
    #     # select hyperparameter and save
    #     hyperparameters = select_hyperparameter(tra_avg, save_path=save_path)

    #     # exog variable
    #     exog_train = get_exog_variable(start_date=tra_avg.index.min(), end_date=tr_end)
    #     exog_test = get_exog_variable(start_date=te_start, end_date=te_end)
    #     exog_pred = get_exog_variable(start_date=pred_start, end_date=pred_end)

    #     # train
    #     vpns = np.unique(subdf["vpn"])
    #     for vpn, _tra, _tes in zip(vpns, tra, tes):
    #         train(vpn, _tra, _tes, exog_train, exog_test, exog_pred, hyperparameters, save_path)

    #     # evaluate (excel report as output)
    #     evaluate(vpns, save_path, info=cluster)
    # else:
    #     logger.info(f"{cluster} is skipped, len(training_period) < 104")
    #     skipped_items = subdf[["vpn"]].unique()
    #     skipped_items["info"] = f"len(training_period) < 104, {cluster}"
    #     skipped_items.to_csv(os.path.join(save_path, "skipped_report.csv"))

# COMMAND ----------

in_sample_forecasts = model.predict_in_sample()

# COMMAND ----------

sum(in_sample_forecasts)

# COMMAND ----------

sum(_tra)

# COMMAND ----------

display(agg_testing_error)

# COMMAND ----------

display(agg_testing_error)

# COMMAND ----------

import matplotlib.pyplot as plt

# COMMAND ----------

for vpn, _tra, _tes in zip(vpns, tra, tes):
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    axs[0].plot(_tra)
    axs[1].plot(_tes)

# COMMAND ----------

display(agg_testing_error)

# COMMAND ----------

tes[0]

# COMMAND ----------

sum(predictions)

# COMMAND ----------

save_path

# COMMAND ----------

for cluster in tqdm(distinct_cluster):
    subdf = sales[sales["cluster"] == cluster]
    save_path = os.path.join(result_path, cluster)
    try:
        evaluate(np.unique(subdf["vpn"]), save_path, info=cluster)
    except Exception as e:
        print(e)

# COMMAND ----------

# concat all reports
results = []
for cluster in tqdm(distinct_cluster):
    try:
        save_path = os.path.join(result_path, cluster)
        result = pd.read_csv(os.path.join(save_path, "model_report.csv"))
        results.append(result)
    except Exception as e:
        print(e)
        result = pd.read_csv(os.path.join(save_path, "skipped_report.csv"))
        results.append(result)

report = pd.concat(results)
report.to_csv(os.path.join(result_path, "model_report.csv"), index=False)

# COMMAND ----------

result

# COMMAND ----------

report

# COMMAND ----------

subdf = sales[sales["cluster"] == "kmean_2023-09-01_irregular_spa_0"]
save_path = os.path.join(result_path, "kmean_2023-09-01_irregular_spa_0")

# COMMAND ----------

vpns = np.unique(subdf["vpn"])
info="kmean_2023-09-01_irregular_spa_0"

# COMMAND ----------

gt_and_pred = []
for vpn in tqdm(vpns, total=len(vpns)):
    encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
    folder = os.path.join(save_path, encoded_vpn)
    tes = pd.read_csv(f"{folder}/dataset_test.csv")
    test = pd.read_csv(f"{folder}/dataset_prediction_on_test.csv")
    test_and_pred = [vpn, sum(tes["qty"]), sum(test["predicted_mean"])]
    gt_and_pred.append(test_and_pred)

agg_testing_error = pd.DataFrame(gt_and_pred, columns=["vpn", "gt", "model_pred"])
agg_testing_error["model_mape (%)"] = abs(agg_testing_error["model_pred"] / (agg_testing_error["gt"]) - 1) * 100

# get sales vel
sales_velocities = {}
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")
    start, end = subdf.index.min(), subdf.index.max()
    date_range = pd.date_range(start, end, freq="W-MON")
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
sales_velocities["sales_vel_pred"] = sales_velocities["weekly_sales"] * len(
    pd.date_range(te_start, te_end, freq="W-MON"))

# join table
result = sales_velocities[["vpn", "sales_vel_pred"]].merge(agg_testing_error, how="left", on="vpn")
result["vel_mape (%)"] = abs(result["sales_vel_pred"] / result["gt"] - 1) * 100
result = result[["vpn", "gt", "sales_vel_pred", "vel_mape (%)", "model_pred", "model_mape (%)"]]
result["info"] = info
# save
result.to_csv(os.path.join(save_path, "model_report.csv"), index=False)
display(result)

# COMMAND ----------

sales_velocities

# COMMAND ----------

start, end

# COMMAND ----------

subdf["order_week"] = subdf.index

# COMMAND ----------

subdf["order_week"].min(), subdf["order_week"].max()

# COMMAND ----------

subdf

# COMMAND ----------


