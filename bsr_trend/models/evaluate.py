# Databricks notebook source
import os
import pandas as pd
from bsr_trend.utils.catalog import BEST_MODEL_REPORT_PATH, TRAINING_DIR


def get_best_model(row):
    d = {"sales_vel": row["vel_mape (%)"], "ets": row["ets_MAPE (%)"], "croston": row["croston_MAPE (%)"]}
    return min(d, key=d.get)


vel_report = pd.read_csv(os.path.join(TRAINING_DIR, "sales_vel", "model_report.csv"))
ets_report = pd.read_csv(os.path.join(TRAINING_DIR, "ets_models", "model_report.csv"))
croston_report = pd.read_csv(os.path.join(TRAINING_DIR, "croston_models", "model_report.csv"))

final_report = pd.merge(
    vel_report,
    ets_report[["vpn", "predicted_qty", "MAPE (%)"]].rename(
        columns={"predicted_qty": "ets_predicted_qty", "MAPE (%)": "ets_MAPE (%)"}
    ),
    how="left",
    on="vpn",
)

final_report = pd.merge(
    final_report,
    croston_report[["vpn", "predicted_qty", "MAPE (%)"]].rename(
        columns={"predicted_qty": "croston_predicted_qty", "MAPE (%)": "croston_MAPE (%)"}
    ),
    how="left",
    on="vpn",
)
final_report = final_report[~final_report["gt"].isna()]
final_report["best_model"] = final_report.apply(lambda row: get_best_model(row), axis=1)
final_report.to_csv(BEST_MODEL_REPORT_PATH, index=False)
