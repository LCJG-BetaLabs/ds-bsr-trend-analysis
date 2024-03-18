import os
import numpy as np
import pandas as pd
from bsr_trend.utils.data import get_time_series
from bsr_trend.logger import get_logger

logger = get_logger()


class TimeSeriesModel:
    def __init__(self, data, tr_start, tr_end, te_start, te_end, fh=None, mode="train"):
        self.data = data
        self.tr_start, self.tr_end = tr_start, tr_end
        self.te_start, self.te_end = te_start, te_end
        self.fh = fh
        self.mode = mode

        self.model = self.init_model()
        self.dirs = self.init_directory()

    def init_model(self):
        raise NotImplementedError

    def init_directory(self):
        raise NotImplementedError

    def train(self, train_data: list) -> None:
        """save trained model"""
        raise NotImplementedError

    def predict(self, fh) -> pd.DataFrame:
        """save dataframe containing vpn, order_week, predicted qty"""
        raise NotImplementedError

    def evaluate(self, test_data: list, vpn: np.ndarray):
        """save dataframe containing vpn, ground_truth, aggregated predicted qty, MAPE (%)"""
        ground_truth = [(vpn, sum(tes[0])) for tes in test_data]
        ground_truth = pd.DataFrame(ground_truth, columns=["vpn", "ground_truth"])
        test_predictions = pd.read_csv(os.path.join(self.dirs, "predictions.csv"))
        test_predictions = test_predictions[["vpn", "predicted_qty"]].groupby("vpn").sum().reset_index()
        result = pd.merge(ground_truth, test_predictions, how="left", on="vpn")
        result["MAPE (%)"] = abs(result["predicted_qty"] / (result["ground_truth"]) - 1) * 100
        result.to_csv(os.path.join(self.dirs, "model_report.csv"))
        logger.info("Model report saved to {}".format(os.path.join(self.dirs, "model_report")))

    def train_predict(self):
        # split data
        if self.mode == "train":
            tra_vpns, tra = get_time_series(self.data, dynamic_start=True, end_date=self.tr_end)
            tes_vpns, tes = get_time_series(self.data, dynamic_start=False, start_date=self.te_start, end_date=self.te_end)
            fh = np.unique(tes["order_week"])
            self.train(tra)
            self.predict(fh)
            self.evaluate(tes, tes_vpns)
        if self.mode == "predict":
            # predict the real future
            if self.fh is None:
                raise ValueError("Forecast horizon must be specified for mode='predict'")
            tra = get_time_series(self.data, dynamic_start=True, end_date=self.te_end)
            self.train(tra)
            self.predict(self.fh)
