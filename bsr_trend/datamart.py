# Databricks notebook source
# MAGIC %md
# MAGIC Filter for BSR:
# MAGIC - u.UDA_ID = 221 AND u.UDA_VALUE = 3 -- is bsr
# MAGIC
# MAGIC Filter for class:
# MAGIC ```
# MAGIC   AND (
# MAGIC     (m.DEPT = 6401 AND m.CLASS = 5)  -- Bath & Towels -> Towels
# MAGIC     OR (m.DEPT = 6401 AND m.CLASS = 4)  -- Bath & Towels -> Bath & Body
# MAGIC     OR (m.DEPT = 6402 AND m.CLASS = 3)  -- Bed Linens -> Linens
# MAGIC     OR (m.DEPT = 6409) -- Home Fragrance & Spa
# MAGIC     or (m.dept=6404) --- Food  and Gourment
# MAGIC   )
# MAGIC ```
# MAGIC
# MAGIC Filter for HL:
# MAGIC - g.GROUP_NAME = "Lifestyle"  -- HL bu

# COMMAND ----------

import datetime
import matplotlib.pyplot as plt
import pandas as pd

from bsr_trend.utils.catalog import (
    write_uc_table,
    SALES,
    WEEK_COVERAGE,
    VPN_STYLE_MAP,
    VPN_INFO,
)

# COMMAND ----------

dbutils.widgets.removeAll()
# format: yyyy-MM-dd
# default: today
dbutils.widgets.text("cutoff_date", datetime.datetime.today().date().strftime("%Y-%m-%d")) 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW BSRStyles AS
# MAGIC SELECT
# MAGIC   COALESCE(m.ITEM_PARENT, m.ITEM) AS style,
# MAGIC   m.BRAND_NAME AS brand_code,
# MAGIC   BRAND_DESCRIPTION as brand_name,
# MAGIC   DEPT,
# MAGIC   class,
# MAGIC   FIRST(m.DESC_UP) AS item_desc
# MAGIC FROM
# MAGIC   lc_prd.neoclone_silver.rms_item_master m
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_uda_item_lov u USING (ITEM) -- group ~ bu
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_groups g ON (INT(FLOOR(m.DEPT / 100) * 100) = g.GROUP_NO)
# MAGIC   left join lc_prd.neoclone_silver.rms_brand b on m.brand_name = b.brand_name
# MAGIC   WHERE
# MAGIC   u.UDA_ID = 221 AND u.UDA_VALUE = 3 -- is bsr
# MAGIC   AND g.GROUP_NAME = "Lifestyle"  -- HL bu
# MAGIC GROUP BY
# MAGIC   style,
# MAGIC   m.BRAND_NAME,
# MAGIC   BRAND_DESCRIPTION,
# MAGIC   dept,
# MAGIC   class;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW ItemVPNMapping AS
# MAGIC SELECT
# MAGIC   s.ITEM AS item,
# MAGIC   s.VPN AS vpn
# MAGIC FROM
# MAGIC   lc_prd.neoclone_silver.rms_item_supplier s
# MAGIC WHERE
# MAGIC   s.VPN IS NOT NULL
# MAGIC   AND s.VPN != "";

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW InStockStyles AS
# MAGIC SELECT
# MAGIC   lcStyleCode AS style,
# MAGIC   stockLevel AS stock_level
# MAGIC FROM
# MAGIC   lc_prd.api_product_feed_silver.lc_product_feed
# MAGIC WHERE
# MAGIC   load_date = (
# MAGIC     SELECT
# MAGIC       MAX(load_date)
# MAGIC     FROM
# MAGIC       lc_prd.api_product_feed_silver.lc_product_feed
# MAGIC   )
# MAGIC   AND stockLevel >= 1
# MAGIC   AND region = "hk";

# COMMAND ----------

# MAGIC %sql
# MAGIC -- items in stock in the past month
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW InStockStylesP1M AS
# MAGIC SELECT
# MAGIC   style,
# MAGIC   COALESCE(stock_level, 0) AS stock_level
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       DISTINCT lcStyleCode AS style
# MAGIC     FROM
# MAGIC       lc_prd.api_product_feed_silver.lc_product_feed
# MAGIC     WHERE
# MAGIC       to_date(load_date, "yyyyMMdd") >= add_months(CURRENT_DATE(), -1) --DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
# MAGIC       AND stockLevel >= 1
# MAGIC       AND region = "hk"
# MAGIC   ) FULL
# MAGIC   JOIN InStockStyles USING (style);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Getting records for past 12 months at weekly level for WOC caluculation***/
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW StyleWeeklySales AS WITH StyleSales AS (
# MAGIC   SELECT
# MAGIC     style,
# MAGIC     DATE_TRUNC("WEEK", TO_DATE(order_date, "yyyyMMdd")) AS order_week,
# MAGIC     dtl_qty,
# MAGIC     amt_hkd
# MAGIC   FROM
# MAGIC     lc_prd.crm_db_neo_silver.dbo_v_sales_dtl
# MAGIC   WHERE
# MAGIC     TO_DATE(order_date, "yyyyMMdd") >= date_add((getArgument("cutoff_date")), -365)
# MAGIC     and TO_DATE(order_date, "yyyyMMdd") <= (getArgument("cutoff_date"))
# MAGIC     AND loc_code IN ("148", "168", "188", "210", "228")
# MAGIC     AND dtl_qty > 0 -- exclude return product
# MAGIC )
# MAGIC SELECT
# MAGIC   style,
# MAGIC   order_week,
# MAGIC   FLOAT(SUM(dtl_qty)) AS qty,
# MAGIC   FLOAT(SUM(amt_hkd)) AS amt
# MAGIC FROM
# MAGIC   StyleSales
# MAGIC GROUP BY
# MAGIC   style,
# MAGIC   order_week;
# MAGIC
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW BSRStyleWeeklySales AS
# MAGIC SELECT
# MAGIC   s.*
# MAGIC FROM
# MAGIC   StyleWeeklySales s
# MAGIC   INNER JOIN BSRStyles b ON (s.style = b.style)
# MAGIC   INNER JOIN InStockStyles i ON (s.style = i.style);
# MAGIC
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW BSRStyleStockLevel AS
# MAGIC SELECT
# MAGIC   i.*
# MAGIC FROM
# MAGIC   InStockStyles i
# MAGIC   INNER JOIN BSRStyles b ON (i.style = b.style);

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY VIEW BSRVPNWeeklySales AS
# MAGIC SELECT
# MAGIC   mp.vpn,
# MAGIC   s.order_week,
# MAGIC   FLOAT(SUM(s.qty)) AS qty,
# MAGIC   FLOAT(SUM(s.amt)) AS amt
# MAGIC FROM
# MAGIC   BSRStyleWeeklySales s
# MAGIC   LEFT JOIN ItemVPNMapping mp ON (s.style = mp.item)
# MAGIC WHERE
# MAGIC   mp.vpn IS NOT NULL
# MAGIC GROUP BY
# MAGIC   mp.vpn,
# MAGIC   s.order_week;
# MAGIC   
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW BSRVPNStockLevel AS
# MAGIC SELECT
# MAGIC   mp.vpn,
# MAGIC   FLOAT(SUM(s.stock_level)) AS stock_level
# MAGIC FROM
# MAGIC   BSRStyleStockLevel s
# MAGIC   LEFT JOIN ItemVPNMapping mp ON (s.style = mp.item)
# MAGIC WHERE
# MAGIC   mp.vpn IS NOT NULL
# MAGIC GROUP BY
# MAGIC   mp.vpn;
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW VPNInfo AS
# MAGIC SELECT
# MAGIC   mp.vpn,
# MAGIC   m.BRAND_NAME AS brand_code,
# MAGIC   BRAND_DESCRIPTION as brand_name,
# MAGIC   FIRST(COALESCE(m.ITEM_PARENT, m.ITEM)) AS style,
# MAGIC   FIRST(m.DESC_UP) AS item_desc,
# MAGIC   FIRST(m.DEPT) AS category,
# MAGIC   FIRST(m.CLASS) AS class,
# MAGIC   FIRST(m.SUBCLASS) AS subclass,
# MAGIC   FIRST(d.DIFF_ID) AS lc_color,
# MAGIC   FIRST(d.DIFF_DESC) AS color_desc,
# MAGIC   FIRST(p.PHASE_DESC) AS phase
# MAGIC FROM
# MAGIC   ItemVPNMapping mp
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_item_master m USING (ITEM)
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_diff_ids d ON (
# MAGIC     m.DIFF_1 = d.DIFF_ID
# MAGIC     AND d.DIFF_TYPE = "COLOUR"
# MAGIC   )
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_item_seasons ris ON (m.ITEM = ris.ITEM)
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_phases p ON (
# MAGIC     ris.SEASON_ID = p.SEASON_ID
# MAGIC     AND ris.PHASE_ID = p.PHASE_ID
# MAGIC   )
# MAGIC   left join lc_prd.neoclone_silver.rms_brand b on m.brand_name = b.brand_name
# MAGIC WHERE
# MAGIC   mp.vpn IS NOT NULL
# MAGIC GROUP BY
# MAGIC   mp.vpn,
# MAGIC   m.BRAND_NAME,
# MAGIC   b.BRAND_DESCRIPTION;
# MAGIC
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW VPNInfoDetail AS
# MAGIC SELECT
# MAGIC   v.vpn,
# MAGIC   brand_code,
# MAGIC   brand_name,
# MAGIC   CONCAT(v.category, "- ", d.DEPT_NAME) AS category,
# MAGIC   CONCAT(v.class, "- ", c.CLASS_NAME) AS class,
# MAGIC   CONCAT(v.subclass, "- ", s.SUB_NAME) AS subclass,
# MAGIC   v.style,
# MAGIC   v.item_desc AS style_desc,
# MAGIC   v.lc_color,
# MAGIC   v.color_desc,
# MAGIC   v.phase
# MAGIC FROM
# MAGIC   VPNInfo v
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_deps d ON (v.category = d.DEPT)
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_class c ON (
# MAGIC     v.category = c.DEPT
# MAGIC     AND v.class = c.CLASS
# MAGIC   )
# MAGIC   LEFT JOIN lc_prd.neoclone_silver.rms_subclass s ON (
# MAGIC     v.category = s.DEPT
# MAGIC     AND v.class = s.CLASS
# MAGIC     AND v.subclass = s.SUBCLASS
# MAGIC   );
# MAGIC
# MAGIC CREATE
# MAGIC   OR REPLACE TEMPORARY VIEW VPNPODetail AS WITH Cte AS (
# MAGIC     SELECT
# MAGIC       -- ol.ORDER_NO,
# MAGIC       -- ol.ITEM,
# MAGIC       mp.VPN AS vpn,
# MAGIC       s.SUP_NAME AS supplier,
# MAGIC       oh.CURRENCY_CODE AS currency,
# MAGIC       FLOAT(ol.UNIT_COST) AS cost,
# MAGIC       FLOAT(ol.UNIT_COST / oh.EXCHANGE_RATE) AS cost_hkd,
# MAGIC       FLOAT(ol.UNIT_RETAIL) AS retail_hkd,
# MAGIC       ROW_NUMBER() OVER(
# MAGIC         PARTITION BY mp.VPN
# MAGIC         ORDER BY
# MAGIC           ol.ORDER_NO DESC
# MAGIC       ) AS rn
# MAGIC     FROM
# MAGIC       lc_prd.neoclone_silver.rms_ordloc ol
# MAGIC       INNER JOIN lc_prd.neoclone_silver.rms_ordhead oh ON (ol.ORDER_NO = oh.ORDER_NO)
# MAGIC       INNER JOIN lc_prd.neoclone_silver.rms_wh wh ON (
# MAGIC         ol.LOCATION = wh.WH
# MAGIC         AND wh.VAT_REGION = 10
# MAGIC       ) -- HK warehouse
# MAGIC       INNER JOIN lc_prd.neoclone_silver.rms_sups s ON (oh.SUPPLIER = s.SUPPLIER)
# MAGIC       INNER JOIN ItemVPNMapping mp ON (ol.ITEM = mp.ITEM)
# MAGIC   )
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   Cte
# MAGIC WHERE
# MAGIC   rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW BSRStylePO AS
# MAGIC WITH Cte AS (
# MAGIC   SELECT
# MAGIC     COALESCE(m.ITEM_PARENT, m.ITEM) AS style,
# MAGIC     o.QTY_RECEIVED AS po_qty
# MAGIC   FROM lc_prd.neoclone_silver.rms_ordloc o
# MAGIC   INNER JOIN lc_prd.neoclone_silver.rms_item_master m ON (o.ITEM = m.ITEM OR o.ITEM = m.ITEM_PARENT)
# MAGIC   WHERE 
# MAGIC     ESTIMATED_INSTOCK_DATE BETWEEN DATE_SUB(CURRENT_DATE(), 6 * 30) AND DATE_SUB(CURRENT_DATE(), 3 * 30)
# MAGIC     AND QTY_RECEIVED IS NOT NULL
# MAGIC )
# MAGIC SELECT
# MAGIC   c.style,
# MAGIC   SUM(c.po_qty) AS po_qty
# MAGIC FROM Cte c
# MAGIC INNER JOIN BSRStyles b ON (c.style = b.style)
# MAGIC GROUP BY c.style;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW BSRVPNPO AS
# MAGIC SELECT
# MAGIC   mp.vpn,
# MAGIC   FLOAT(SUM(s.po_qty)) AS po_qty
# MAGIC FROM BSRStylePO s
# MAGIC LEFT JOIN ItemVPNMapping mp ON (s.style = mp.item)
# MAGIC WHERE mp.vpn IS NOT NULL
# MAGIC GROUP BY mp.vpn;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW VPNPODetail AS
# MAGIC WITH Cte AS (
# MAGIC   SELECT 
# MAGIC     -- ol.ORDER_NO,
# MAGIC     -- ol.ITEM,
# MAGIC     mp.VPN AS vpn,
# MAGIC     s.SUP_NAME AS supplier,
# MAGIC     oh.CURRENCY_CODE AS currency,
# MAGIC     FLOAT(ol.UNIT_COST) AS cost,
# MAGIC     FLOAT(ol.UNIT_COST / oh.EXCHANGE_RATE) AS cost_hkd,
# MAGIC     FLOAT(ol.UNIT_RETAIL) AS retail_hkd,
# MAGIC     ROW_NUMBER() OVER(PARTITION BY mp.VPN ORDER BY ol.ORDER_NO DESC) AS rn
# MAGIC   FROM lc_prd.neoclone_silver.rms_ordloc ol
# MAGIC   INNER JOIN lc_prd.neoclone_silver.rms_ordhead oh ON (ol.ORDER_NO = oh.ORDER_NO)
# MAGIC   INNER JOIN lc_prd.neoclone_silver.rms_wh wh ON (ol.LOCATION = wh.WH AND wh.VAT_REGION = 10) -- HK warehouse
# MAGIC   INNER JOIN lc_prd.neoclone_silver.rms_sups s ON (oh.SUPPLIER = s.SUPPLIER)
# MAGIC   INNER JOIN ItemVPNMapping mp ON (ol.ITEM = mp.ITEM)
# MAGIC )
# MAGIC SELECT * FROM Cte WHERE rn = 1;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW BSRVPNStyleMapping AS
# MAGIC SELECT
# MAGIC   DISTINCT mp.vpn,
# MAGIC   s.style
# MAGIC FROM BSRStyleWeeklySales s
# MAGIC LEFT JOIN ItemVPNMapping mp ON (s.style = mp.item)
# MAGIC WHERE mp.vpn IS NOT NULL;

# COMMAND ----------

sales = spark.table("BSRVPNWeeklySales").toPandas()
stock_level = spark.table("BSRVPNStockLevel").toPandas()
po_qty = spark.table("BSRVPNPO").toPandas()
vpn_info = spark.table("VPNInfoDetail").toPandas()
vpn_po = spark.table("VPNPODetail").toPandas()

# COMMAND ----------

# Use past 3 months average as the average weekly sales per item
today_week = datetime.datetime.today() - datetime.timedelta(days=datetime.date.today().weekday())

avg_weekly_sales = sales[sales["order_week"] >= (today_week - datetime.timedelta(days=30*3))].groupby("vpn")["qty"].sum() / 12
avg_weekly_sales = avg_weekly_sales.to_frame().rename(columns={"qty": "weekly_sales"})
avg_weekly_sales

coverage = pd.merge(
    avg_weekly_sales,
    po_qty.set_index("vpn"),
    left_index=True,
    right_index=True,
    how="left",
)
coverage = pd.merge(
    coverage,
    stock_level.set_index("vpn"),
    left_index=True,
    right_index=True,
    how="left",
)
coverage["weeks_coverage"] = coverage["stock_level"] / coverage["weekly_sales"]
coverage = pd.merge(coverage.reset_index(), vpn_info, how="left", on="vpn")
coverage = pd.merge(coverage, vpn_po, how="left", on="vpn")
coverage = coverage[[
    "supplier",
    "category",
    "class",
    "subclass",
    "vpn",
    "style",
    "lc_color",
    "color_desc",
    "style_desc",
    "phase",
    "currency",
    "cost",
    "cost_hkd",
    "retail_hkd",
    "weekly_sales",
    "po_qty",
    "stock_level",
    "weeks_coverage",
]]

# COMMAND ----------

# save to uc
write_uc_table(
    SALES,
    sales,
    mode="overwrite",
)
write_uc_table(
    WEEK_COVERAGE,
    coverage,
    mode="overwrite",
)
write_uc_table(
    VPN_STYLE_MAP,
    spark.table("BSRVPNStyleMapping"),
    mode="overwrite",
)
write_uc_table(
    VPN_INFO,
    vpn_info,
    mode="overwrite",
)

# COMMAND ----------


