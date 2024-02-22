# Databricks notebook source
import pandas as pd
import numpy as np

from bsr_trend.logger import get_logger

logger = get_logger()

# COMMAND ----------

# option -> dynamic_start

# COMMAND ----------

# function to get ts data from table
sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")


def get_time_series(sales, dynamic_start=True, start_date=None, end_date=None):
    sales["order_week"] = pd.to_datetime(sales["order_week"])
    vpns = np.unique(sales["vpn"])
    logger.info(f"Number of VPN: {len(vpns)}")

    if end_date is None:
        raise ValueError(f"end_date need to be specified")

    time_series = []
    if not dynamic_start:
        # fill 0 so that the length of each series is the same
        start, end = sales["order_week"].min(), sales["order_week"].max()
        date_range = pd.date_range(start, end, freq="W-MON")

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

            # take period
            if start_date is None:
                start_date = start
            tra = buf['qty'][start:end_date].dropna()
            tra.sort_index(inplace=True)
            time_series.append(list(tra))
    else:
        # start date of each time series is the first purchase date
        # dynamic start date for each product
        for vpn in vpns:
            subdf = sales[sales["vpn"] == vpn]
            start, end = subdf["order_week"].min(), subdf["order_week"].max()
            subdf = subdf.set_index("order_week")
            date_range = pd.date_range(start, end, freq="W-MON")

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

            # take period
            tra = buf['qty'][start:end_date].dropna()
            tra.sort_index(inplace=True)
            time_series.append(list(tra))
    return time_series

# COMMAND ----------

tr_end = '2023-09-01'
tra = get_time_series(sales, dynamic_start=True, start_date=None, end_date=tr_end)

# COMMAND ----------

def average_demand_interval(time_series):
    """
    Average Demand Interval (ADI)
    Args:
        time_series: list of multiple time series
    return:
        list of ADI
    """
    adi_list = []
    for ts in time_series:
        num_non_zero_demand = len([qty for qty in ts if qty > 0])
        adi = len(ts) / num_non_zero_demand if num_non_zero_demand > 0 else -1
        adi_list.append(adi)
    return np.array(adi_list)

# COMMAND ----------

def coefficient_of_variation(time_series):
    """non-zero demand variation coefficient for demand stability OR Coefficient of Variation (COV^2)"""
    cov_list = []
    for ts in time_series:
        cov = np.std(ts) / np.mean(ts) if len(ts) > 0 and np.mean(ts) > 0 else -1
        cov_list.append(cov)
    return np.array(cov_list)

# COMMAND ----------

def classify_trend(time_series):
    adi = average_demand_interval(time_series)
    cov2 = np.square(coefficient_of_variation(time_series))
    result = np.empty_like(adi, dtype='<U20')
    result[np.logical_and(adi < 1.32, cov2 >= 0.49)] = "irregular"
    result[np.logical_and(adi < 1.32, cov2 < 0.49)] = "smooth"
    result[np.logical_and(adi >= 1.32, cov2 >= 0.49)] = "lumpy"
    result[np.logical_and(adi >= 1.32, cov2 < 0.49)] = "intermittent"
    result[np.logical_or(adi == -1, cov2 == -1)] = "zero_ts"
    return result

# COMMAND ----------

result = classify_trend(tra)

# COMMAND ----------

np.unique(result)

# COMMAND ----------


