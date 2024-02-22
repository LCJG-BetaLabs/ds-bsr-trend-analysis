import numpy as np
import pandas as pd
from bsr_trend.logger import get_logger

logger = get_logger()


def get_time_series(sales, dynamic_start=True, start_date=None, end_date=None):
    """get ts data from table"""
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
