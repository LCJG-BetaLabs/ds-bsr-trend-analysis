# Databricks notebook source
# MAGIC %run utils/sendgrid_utils

# COMMAND ----------

from bsr_trend.utils.catalog import CUTOFF_DATE, BEST_MODEL_REPORT_PATH

# COMMAND ----------

send_email(
    list_email_to=["arnabmaulik@lcjgroup.com", "cintiaching@lcjgroup.com"],
    str_subject="LC HL BSR Prediction Result",
    str_html_content="The Prediction Result and model report are attached. Thanks.",
    attachments=[f"/dbfs/mnt/prd/bsr_trend/bsr_prediction_{CUTOFF_DATE}.csv", BEST_MODEL_REPORT_PATH]
)

# COMMAND ----------


