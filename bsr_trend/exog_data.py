# Databricks notebook source
# add new exog variables
# holiday tag
# sales period
# traffic data
# price

# COMMAND ----------

# MAGIC %md
# MAGIC holiday tag

# COMMAND ----------

import holidays 

def tag_holidays(date):
    holiday_dates = list(holidays.HK(years=date.year).keys())
    if date in holiday_dates:
        return 1
    else:
        return 0

# COMMAND ----------

# MAGIC %md
# MAGIC traffic data (aggregate to weekly)

# COMMAND ----------

# lc_prd.dashboard_core_kpi_gold.traffic_fact from workflow
# https://adb-2705545515885439.19.azuredatabricks.net/?o=2705545515885439#job/219640479773386/run/742429623685846
# remove dependency when in staging
weekly_traffic = spark.sql(
    """
    SELECT
        date_trunc('week', traffic_date) AS week_start_date,
        SUM(no_of_traffic) AS weekly_traffic
    FROM
        lc_prd.dashboard_core_kpi_gold.traffic_fact
    GROUP BY
        date_trunc('week', traffic_date)
    ORDER BY
        week_start_date
    """
)

# COMMAND ----------

weekly_traffic.write.parquet("/mnt/dev/bsr_trend/exog_data/weekly_traffic.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC sales period

# COMMAND ----------

# hardcode (dec and jun are sales period)
from datetime import datetime

def sales_period(d):
    if d.month in [6, 12]:
        return 1
    else:
        return 0

# COMMAND ----------

# MAGIC %md
# MAGIC price

# COMMAND ----------

# question: we are able to get prices from the past, but not the future (assume to be same as the latest price?)
# use load_date as the day of prices

# COMMAND ----------

import pandas as pd 

mapping_table = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_style_map.csv")


def get_style_code(vpn, mapping_table):
    if not mapping_table[mapping_table["vpn"] == vpn].empty:
        return mapping_table.loc[mapping_table["vpn"] == vpn, "style"].iloc[0]
    else:
        return None


def get_weekly_prices(vpn, start_date, end_date):
    style = get_style_code(vpn)
    prices = spark.sql(
        f"""
        SELECT 
            AVG(price), 
            date_trunc('week', load_date) AS week_start_date,
        FROM lc_prd.api_product_feed_silver.lc_product_feed
        WHERE 
            lcStyleCode = '{style}'
            AND region = "hk"
            AND load_date >= {start_date}
            AND load_date <= {end_date}
        GROUP BY
            date_trunc('week', load_date)
        """
    ).toPandas()
    return prices

# COMMAND ----------


