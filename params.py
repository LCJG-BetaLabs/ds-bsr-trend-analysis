# Databricks notebook source
import datetime

# COMMAND ----------

dbutils.widgets.removeAll()
# format: yyyy-MM-dd
# default: today
dbutils.widgets.text("cutoff_date", datetime.datetime.today().date().strftime("%Y-%m-%d")) 

# COMMAND ----------

with open("/Volumes/lc_prd/ml_trend_analysis_silver/models/latest_cutoff_date.txt", "w") as f:
    f.write(getArgument("cutoff_date"))
