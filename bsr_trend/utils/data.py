import numpy as np
import pandas as pd
from bsr_trend.logger import get_logger
from bsr_trend.utils.catalog import SALES, VPN_INFO
from databricks.sdk.runtime import spark

logger = get_logger()


def get_sales_table() -> pd.DataFrame:
    sales = spark.table(SALES).toPandas()
    # dev: take spa for testing
    vpn_info = spark.table(VPN_INFO).toPandas()
    sales = sales.merge(vpn_info[["vpn", "category"]], how="left", on="vpn")
    sales = sales[sales["category"] == '6409- Home Fragrance & Spa']
    sales = sales.drop("category", axis=1)
    logger.info("Retrieved sales table with Home Fragrance & Spa only")
    return sales


def get_time_series(sales, dynamic_start=True, start_date=None, end_date=None):
    """get ts data from table"""
    sales["order_week"] = pd.to_datetime(sales["order_week"])
    vpns = np.unique(sales["vpn"])
    logger.info(f"Number of VPN: {len(vpns)}")

    if end_date is None:
        raise ValueError(f"end_date must be specified")

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
            buf = buf[["order_week", "qty"]]
            buf = buf.fillna(0).astype(int)

            # take period
            if start_date is None:
                start_date = start
                logger.info(f"""Start date is not specified, taking {start}, the minimum date from sales table.""")
            tra = buf["qty"][start_date:end_date].dropna()
            tra.sort_index(inplace=True)
            time_series.append(tra)
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
            buf = buf[["order_week", "qty"]]
            buf = buf.fillna(0).astype(int)

            # take period
            tra = buf["qty"][start:end_date].dropna()
            tra.sort_index(inplace=True)
            time_series.append(tra)
    return time_series
