import os
import numpy as np
import itertools
import statsmodels.api as sm
import statsmodels
from bsr_trend.logger import get_logger

logger = get_logger()

# cluster averages
path = "/dbfs/mnt/dev/bsr_trend/clustering/clustering_result/"
cluster_averages = np.load(os.path.join(path, "cluster_averages.npy"), allow_pickle=True)


def get_cluster_average(cluster_num):
    return cluster_averages[cluster_num - 1][0]


def extract_season_component(ts, period):
    decomposition = sm.tsa.seasonal_decompose(ts, period=period)
    season_component = decomposition.seasonal
    return season_component


# Assuming a seasonal pattern repeating every 52 weeks (annual seasonality) -> period=52

def choose_d(tra, d=0, alpha=0.05, seasonal=False, period=52):
    # Augmented Dickey-Fuller test to test for stationary
    # p-value is less than 0.05 -> reject H0, take this series stationary
    # else, increase d until stationary
    if seasonal is True:
        tra = extract_season_component(tra, period=period)
    result = sm.tsa.stattools.adfuller(tra)
    p_value = result[1]
    logger.info(f"p-value: {p_value}, d: {d}")
    if p_value < alpha:
        # If p-value is less than 0.05, reject the null hypothesis
        # and conclude that the series is stationary
        return d
    else:
        # If p-value is greater than or equal to 0.05, increase d by 1
        d += 1
        return choose_d(tra.diff().dropna(), d)


def choose_p_and_q(tra, d):
    p_values = range(1, 6)
    q_values = range(1, 6)
    best_aic = float('inf')
    best_p = 0
    best_q = 0

    # Generate all combinations of p and q
    param_grid = itertools.product(p_values, q_values)
    logger.info(f"param_grid: {list(param_grid)}")
    # Perform grid search
    for param in itertools.product(p_values, q_values):
        p, q = param
        model = sm.tsa.ARIMA(tra, order=(p, d, q))
        results = model.fit()
        aic = results.aic

        # Update best AIC and parameters if lower AIC is found
        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_q = q
        logger.info(f"Current AIC: {aic}, p: {p}, q: {q}")
        logger.info(f"Best AIC: {best_aic}, p: {best_p}, q: {best_q}")

    return best_p, best_q


def choose_seasonal_p_and_q(tra, D, s=52, order=(1, 0, 1)):
    p_values, q_values = range(1, 6), range(1, 6)
    best_aic = float('inf')
    best_p, best_q = 0, 0

    # Generate all combinations of p and q
    param_grid = itertools.product(p_values, q_values)
    logger.info(f"param_grid: {list(param_grid)}")

    # Perform grid search
    for param in itertools.product(p_values, q_values):
        p, q = param
        model = statsmodels.tsa.statespace.sarimax.SARIMAX(tra, order=order, seasonal_order=(p, D, q, s), exog=None)
        results = model.fit()
        aic = results.aic

        # Update best AIC and parameters if lower AIC is found
        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_q = q
        logger.info(f"Current AIC: {aic}, P: {p}, Q: {q}")
        logger.info(f"Best AIC: {best_aic}, P: {best_p}, Q: {best_q}")

    return best_p, best_q