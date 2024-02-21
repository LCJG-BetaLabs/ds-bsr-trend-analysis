# Databricks notebook source
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# COMMAND ----------

# compare 3 clustering method
clustering_method = ["som", "kmeans_breakdown", "kmeans"]

all_result = []
for cm in clustering_method:
    dfs = []
    path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
    mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
    for cluster in set(mapping["cluster"].values):
        try:
            df = pd.read_csv(os.path.join(path, "sarimax_result", f"cluster_{cluster}", "agg_result.csv"))
            dfs.append(df)
        except:
            print(cm, f"cluster_{cluster}")
    result = pd.concat(dfs)
    all_result.append(result)

# COMMAND ----------

som, kmeans_breakdown, kmeans = all_result # display(all_result[0])

# COMMAND ----------

# get all vpn
eval_result = None
for cm in clustering_method:
    path = f"/dbfs/mnt/dev/bsr_trend/clustering/{cm}/"
    mapping = pd.read_csv(os.path.join(path, "cluster_mapping.csv"))
    mapping = mapping.rename(columns={"cluster": f"{cm}_cluster_no"})
    if eval_result is None:
        eval_result = mapping
    else:
        eval_result = eval_result.merge(mapping, on="vpn", how="left")

# COMMAND ----------

# get sales in spa
sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
sales["order_week"] = pd.to_datetime(sales["order_week"])

vpn_info = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_info.csv")
sales = sales.merge(vpn_info[["vpn", "category"]], how="left", on="vpn")
# clustering by category
# dev: take spa for testing
sales = sales[sales["category"] == '6409- Home Fragrance & Spa']

# COMMAND ----------

# get all sales vel
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

vpns = np.unique(sales["vpn"])

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

vel_pred = pd.merge(sales_velocities, pred_sales_velocities, how="inner", on="vpn")
vel_pred = vel_pred.drop(columns="weekly_sales")
vel_pred["vel_mape (%)"] = abs(vel_pred["sales_vel_pred"] / (vel_pred["gt"] + 0.01) - 1) * 100

display(vel_pred)

# COMMAND ----------

eval_result = eval_result.merge(vel_pred, on="vpn", how="left")

# COMMAND ----------

# num of week in training period
# num of week in training period that has 0 sales
# num of week in testing period that has 0 sales

num_weeks = []
tra_num_weeks_all = []
tra_num_weeks = []
te_num_weeks = []
for vpn in tqdm(vpns, total=len(vpns)):
    subdf = sales[sales["vpn"] == vpn].set_index("order_week")

    subdf["order_week"] = subdf.index
    start, end = subdf["order_week"].min(), subdf["order_week"].max()

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
    num_week = buf['order_week'].nunique()
    num_weeks.append(num_week)

    buf1 = buf[(buf["order_week"] >= start) & (buf["order_week"] <= tr_end)]
    tra_num_weeks_all.append(buf1['order_week'].nunique())
    num_week = buf1[buf1["qty"] > 0]['order_week'].nunique()
    tra_num_weeks.append(num_week)

    buf2 = buf[(buf["order_week"] >= te_start) & (buf["order_week"] <= te_end)]
    num_week = buf2[buf2["qty"] > 0]['order_week'].nunique()
    te_num_weeks.append(num_week)

# COMMAND ----------

week_info = pd.DataFrame(zip(vpns, num_weeks, tra_num_weeks_all, tra_num_weeks, te_num_weeks), columns=[
    "vpn", "total_num_weeks", "total_train_week", "num_>0_week_train", "num_>0_week_test"])
week_info

# COMMAND ----------

eval_result = eval_result.merge(week_info, on="vpn", how="left")

# COMMAND ----------

# left join result (rename column?)
# "sales_vel_pred", "vel_mape (%)",
som_ = som[["vpn", "model_pred", "model_mape (%)"]].rename(columns={"model_pred": "som_model_pred", "model_mape (%)": "som_model_mape (%)"})

# COMMAND ----------

kmeans_ = kmeans[["vpn", "model_pred", "model_mape (%)"]].rename(columns={"model_pred": "km_model_pred", "model_mape (%)": "km_model_mape (%)"})

# COMMAND ----------

kmeans_breakdown_ = kmeans_breakdown[["vpn", "model_pred", "model_mape (%)"]].rename(columns={"model_pred": "kmb_model_pred", "model_mape (%)": "kmb_model_mape (%)"})

# COMMAND ----------

eval_result = eval_result.merge(som_, on="vpn", how="left")
eval_result = eval_result.merge(kmeans_, on="vpn", how="left")
eval_result = eval_result.merge(kmeans_breakdown_, on="vpn", how="left")

# COMMAND ----------

# best model column

def best_model(row):
    # Get the values of the four columns and their corresponding names
    columns = ['vel_mape (%)', 'som_model_mape (%)', 'km_model_mape (%)', 'kmb_model_mape (%)']
    values = row[columns].values
    names = row[columns].keys()
    
    # Filter out null values and corresponding names
    non_null_values = []
    non_null_names = []
    for value, name in zip(values, names):
        if pd.notnull(value):
            non_null_values.append(value)
            non_null_names.append(name)
    
    # Check if any non-null values exist
    if len(non_null_values) > 0:
        # Find the index of the lowest value
        min_index = np.argmin(non_null_values)
        
        # Return the corresponding name
        return non_null_names[min_index].split("_")[0]
    else:
        return np.nan

# COMMAND ----------

eval_result["best_model"] = eval_result.apply(lambda row: best_model(row), axis=1)

# COMMAND ----------

display(eval_result)

# COMMAND ----------

np.unique(eval_result["best_model"], return_counts=True)

# COMMAND ----------

from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(eval_result["gt"], eval_result["sales_vel_pred"])

# COMMAND ----------

mean_absolute_percentage_error(eval_result[~eval_result["som_model_pred"].isna()]["gt"], eval_result[~eval_result["som_model_pred"].isna()]["som_model_pred"])

# COMMAND ----------

len(eval_result[~eval_result["som_model_pred"].isna()])

# COMMAND ----------

mean_absolute_percentage_error(eval_result[~eval_result["kmb_model_pred"].isna()]["gt"], eval_result[~eval_result["kmb_model_pred"].isna()]["kmb_model_pred"])

# COMMAND ----------

len(eval_result[~eval_result["kmb_model_pred"].isna()])

# COMMAND ----------

mean_absolute_percentage_error(eval_result[~eval_result["km_model_pred"].isna()]["gt"], eval_result[~eval_result["km_model_pred"].isna()]["km_model_pred"])

# COMMAND ----------


