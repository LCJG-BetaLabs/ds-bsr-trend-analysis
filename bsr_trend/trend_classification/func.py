import numpy as np
from bsr_trend.logger import get_logger

logger = get_logger()


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


def coefficient_of_variation(time_series):
    """non-zero demand variation coefficient for demand stability OR Coefficient of Variation (COV^2)"""
    cov_list = []
    for ts in time_series:
        cov = np.std(ts) / np.mean(ts) if len(ts) > 0 and np.mean(ts) > 0 else -1
        cov_list.append(cov)
    return np.array(cov_list)


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


def classify_by_history(time_series):
    result = []
    for ts in time_series:
        if len(ts) >= 24:
            result.append("history>=6months")
        elif len(ts) >= 12 and len(ts) < 24:
            result.append("3months<=history<6months")
        else:
            result.append("history<3months")
    return result
