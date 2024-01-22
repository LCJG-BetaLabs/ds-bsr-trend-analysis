from databricks.sdk.runtime import spark
import pandas as pd
import holidays

# VPN and style code mapping
mapping_table = pd.read_csv("/dbfs/mnt/dev/bsr_trend/vpn_style_map.csv")


def get_weekly_traffic():
    """
    get traffic data (aggregate to weekly)
    source: lc_prd.dashboard_core_kpi_gold.traffic_fact from workflow
    https://adb-2705545515885439.19.azuredatabricks.net/?o=2705545515885439#job/219640479773386/run/742429623685846
    remove dependency when in production
    """
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
    return weekly_traffic


def get_daily_traffic():
    """
    get traffic data (aggregate to daily, all store)
    source: lc_prd.dashboard_core_kpi_gold.traffic_fact from workflow
    https://adb-2705545515885439.19.azuredatabricks.net/?o=2705545515885439#job/219640479773386/run/742429623685846
    remove dependency when in production
    """
    daily_traffic = spark.sql(
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
    return daily_traffic


def tag_holidays(date):
    """holiday tag"""
    holiday_dates = list(holidays.HK(years=date.year).keys())
    if date in holiday_dates:
        return 1
    else:
        return 0


def sales_period(date):
    """hardcode (dec and jun are sales period)"""
    if date.month in [6, 12]:
        return 1
    else:
        return 0


def get_style_code(vpn):
    if not mapping_table[mapping_table["vpn"] == vpn].empty:
        return mapping_table.loc[mapping_table["vpn"] == vpn, "style"].iloc[0]
    else:
        return None


def get_weekly_prices(vpn, start_date, end_date):
    """
    question: we are able to get prices from the past, but not the future (assume to be same as the latest price?)
    use load_date as the day of prices
    """
    style = get_style_code(vpn)
    prices = spark.sql(
        f"""
        SELECT 
            AVG(price) AS avg_price, 
            DATE(date_trunc('week', to_date(load_date, "yyyyMMdd"))) AS order_week
        FROM lc_prd.api_product_feed_silver.lc_product_feed
        WHERE 
            lcStyleCode = {style}
            AND region = "hk"
            AND load_date >= {start_date}
            AND load_date <= {end_date}
        GROUP BY 
            DATE(date_trunc('week', to_date(load_date, "yyyyMMdd")))
        """
    ).toPandas()
    return prices


def one_hot_encode_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[date_col].apply(lambda x: x.month)
    month_dummies = pd.get_dummies(df['month'], prefix="month", prefix_sep="-")
    for i in range(1, 13):
        col = f"month-{i}"
        if col not in month_dummies.columns:
            month_dummies[col] = 0
    month_dummies = month_dummies[[f"month-{i}" for i in range(1, 13)]]
    df = df.drop(columns=[c for c in month_dummies.columns if c in df.columns])
    df = pd.concat([df, month_dummies], axis=1).drop(['month'], axis=1)
    return df
