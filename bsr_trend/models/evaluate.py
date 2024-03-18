# Databricks notebook source
dbutils.widgets.removeAll()

dbutils.widgets.text("run_date", "20240306")
run_date = getArgument("run_date")

# COMMAND ----------

import pandas as pd
from bsr_trend.utils.catalog import TREND_CLASSIFICATION_RESULT, VPN_INFO

# COMMAND ----------

def get_best_model(row):
    d = {"sales_vel": row["vel_mape (%)"], "arima": row["arima_model_mape (%)"], "ets": row["ets_model_mape (%)"], "croston": row["croston_model_mape (%)"]}
    return min(d, key=d.get)

# COMMAND ----------

vpn_info = spark.table(VPN_INFO).toPandas()
trend_class = spark.sql("SELECT * FROM lc_dev.ml_trend_analysis_silver.trend_classification").toPandas()

# COMMAND ----------

vel_report = pd.read_csv(f"/dbfs/mnt/dev/bsr_trend/sales_vel_{run_date}/model_report.csv")
arima_report = pd.read_csv(f"/dbfs/mnt/dev/bsr_trend/arima_result_{run_date}/model_report.csv")
ets_report = pd.read_csv(f"/dbfs/mnt/dev/bsr_trend/ets_model_{run_date}/model_report.csv")
croston_report = pd.read_csv(f"/dbfs/mnt/dev/bsr_trend/crostons_method_{run_date}/model_report.csv")

# COMMAND ----------

final_report = pd.merge(
    vel_report, 
    arima_report[["vpn", "model_pred", "model_mape (%)"]].rename(
        columns={"model_pred": "arima_model_pred", "model_mape (%)": "arima_model_mape (%)"}
    ),
    how="left",
    on="vpn",
)

final_report = pd.merge(
    final_report,
    ets_report[["vpn", "model_pred", "model_mape (%)"]].rename(
        columns={"model_pred": "ets_model_pred", "model_mape (%)": "ets_model_mape (%)"}
    ),
    how="left",
    on="vpn",
)

final_report = pd.merge(
    final_report,
    croston_report[["vpn", "model_pred", "model_mape (%)"]].rename(
        columns={"model_pred": "croston_model_pred", "model_mape (%)": "croston_model_mape (%)"}
    ),
    how="left",
    on="vpn",
)
final_report = pd.merge(trend_class, final_report, how="left", on="vpn")
final_report = final_report[~final_report["gt"].isna()]

# COMMAND ----------

final_report["best_model"] = final_report.apply(lambda row: get_best_model(row), axis=1)
final_report = final_report.merge(vpn_info[["vpn", "category"]], on="vpn", how="left")
final_report.to_csv("/dbfs/mnt/dev/bsr_trend/final_model_report_all_class.csv", index=False)
