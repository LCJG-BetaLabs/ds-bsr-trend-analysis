# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# COMMAND ----------

sales = pd.read_csv("/dbfs/mnt/dev/bsr_trend/sales.csv")
sales["order_week"] = pd.to_datetime(sales["order_week"])

# COMMAND ----------

start, end = sales["order_week"].min(), sales["order_week"].max()
date_range = pd.date_range(start, end, freq="W-MON")

# COMMAND ----------

sales

# COMMAND ----------

sales.groupby("vpn")["qty"].sum().sort_values()

# COMMAND ----------

sales.groupby("order_week").agg({"qty": "sum"}).plot()
plt.title("Overall sales (qty) per week")
plt.show()

# COMMAND ----------

query_vpn = "KEBOUFLMIHE"

subdf = sales[sales["vpn"] == query_vpn].set_index("order_week")["qty"].sort_index().to_frame()
filled_ts = pd.merge(
    pd.DataFrame(index=date_range),
    subdf,
    how="left",
    left_index=True,
    right_index=True,
).fillna(0).astype(int)
filled_ts.plot()
plt.title(f"Weekly sales (vpn={query_vpn})")
plt.show()

# COMMAND ----------

res = sm.tsa.seasonal_decompose(filled_ts["qty"].dropna())
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

# COMMAND ----------

query_vpn = "B500BT13RF"

subdf = sales[sales["vpn"] == query_vpn].set_index("order_week")["qty"].sort_index().to_frame()
filled_ts = pd.merge(
    pd.DataFrame(index=date_range),
    subdf,
    how="left",
    left_index=True,
    right_index=True,
).fillna(0).astype(int)
filled_ts.plot()
plt.title(f"Weekly sales (vpn={query_vpn})")
plt.show()

# COMMAND ----------

buf = filled_ts
buf

# COMMAND ----------

res = sm.tsa.seasonal_decompose(buf["qty"].dropna())
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()

# COMMAND ----------

#train_test_split
tr_start, tr_end = '2019-12-30', '2023-05-01'
te_start, te_end = '2023-05-08', '2023-11-06'
tra = buf['qty'][tr_start:tr_end].dropna()
tes = buf['qty'][te_start:te_end].dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC # model choice

# COMMAND ----------

#ADF-test(Original-time-series)
res = sm.tsa.adfuller(buf['qty'].dropna(), regression='ct')
print('p-value:{}'.format(res[1]))

# COMMAND ----------

#ADF-test(differenced-time-series)
res = sm.tsa.adfuller(buf['qty'].diff().dropna(),regression='c')
print('p-value:{}'.format(res[1]))

# COMMAND ----------

#we use tra.diff()(differenced data), because this time series is unit root process.
fig,ax = plt.subplots(2,1,figsize=(20,10))
fig = sm.graphics.tsa.plot_acf(tra.diff().dropna(), lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(tra.diff().dropna(), lags=50, ax=ax[1])
plt.show()

# COMMAND ----------

resDiff = sm.tsa.arma_order_select_ic(tra, max_ar=7, max_ma=7, ic='aic', trend='c')
ar, ma = resDiff["aic_min_order"]
print(f"ARMA(p, q) = ({ar}, {ma}) is best")

# COMMAND ----------

# MAGIC %md # ARIMAX

# COMMAND ----------

print(f"Training on ({ar}, 1, {ma})")
arima = sm.tsa.statespace.SARIMAX(
    tra,
    order=(ar, 1, ma), 
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False, 
    enforce_invertibility=False,
).fit()
arima.summary()
#We can use SARIMAX model as ARIMAX when seasonal_order is (0,0,0,0) .

# COMMAND ----------

res = arima.resid
fig, ax = plt.subplots(2,1,figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()

# COMMAND ----------

pred = arima.predict(tr_end, te_end)[1:]
print('ARIMA model MSE:{}'.format(mean_squared_error(tes, pred)))
print('ARIMA model MAPEE:{}'.format(mean_absolute_percentage_error(tes, pred)))

# COMMAND ----------

pd.DataFrame({'test':tes,'pred':pred}).plot();plt.show()

# COMMAND ----------

# MAGIC %md # SARIMAX

# COMMAND ----------

print(f"Training on ({ar}, 1, {ma})")
sarima = sm.tsa.statespace.SARIMAX(
    tra,
    order=(ar, 1, ma), 
    seasonal_order=(7, 1, 7, 7),
    enforce_stationarity=True, 
    enforce_invertibility=False,
).fit()
sarima.summary()

# COMMAND ----------

res = sarima.resid
fig, ax = plt.subplots(2, 1, figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()

# COMMAND ----------

pred = arima.predict(tr_end, te_end)[1:]
print('ARIMA model MSE:{}'.format(mean_squared_error(tes, pred)))
print('ARIMA model MAPEE:{}'.format(mean_absolute_percentage_error(tes, pred)))

# COMMAND ----------

pd.DataFrame({'test':tes,'pred':pred}).plot();plt.show()

# COMMAND ----------

