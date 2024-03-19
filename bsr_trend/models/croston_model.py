import os
import shutil
import base64
import numpy as np
import pandas as pd
from sktime.forecasting.croston import Croston
from sktime.utils import mlflow_sktime

from bsr_trend.models.model import TimeSeriesModel
from bsr_trend.utils.catalog import TRAINING_DIR, PREDICTION_DIR, IS_DATABRICKS


class CrostonModel(TimeSeriesModel):
    @staticmethod
    def init_model():
        croston_model = Croston()
        return croston_model

    def init_directory(self):
        if self.mode == "train":
            directory = os.path.join(TRAINING_DIR, "croston_models")
        elif self.mode == "predict":
            directory = os.path.join(PREDICTION_DIR, "croston_models")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def train(self, train_data: list, vpns: np.ndarray) -> None:
        """save trained model"""
        for tra, vpn in zip(train_data, vpns):
            forecaster = self.init_model()
            forecaster.fit(tra)
            # encode vpn
            encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
            folder = os.path.join(self.dir, encoded_vpn)
            os.makedirs(folder, exist_ok=True)
            # save model
            if IS_DATABRICKS:
                # OSError: [Errno 95] Operation not supported for writing zip file directly to volume
                # saving to tmp then move to volume
                tmp_path = "/tmp/bsr_trend/model"
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
                mlflow_sktime.save_model(
                    sktime_model=forecaster,
                    path=tmp_path
                )
                if os.path.exists(os.path.join(folder, "model")):
                    shutil.rmtree(os.path.join(folder, "model"))
                shutil.move(tmp_path, folder)
            else:
                mlflow_sktime.save_model(
                    sktime_model=forecaster,
                    path=folder,
                )

    def predict(self, fh: int, vpns: np.ndarray) -> None:
        """save dataframe containing vpn, order_week, predicted qty"""
        # load model
        all_predictions = []
        for vpn in vpns:
            encoded_vpn = base64.b64encode(vpn.encode("utf-8")).decode()
            folder = os.path.join(self.dir, encoded_vpn)
            loaded_model = mlflow_sktime.load_model(model_uri=os.path.join(folder, "model"))
            # predict
            pred = loaded_model.predict(fh=range(1, fh))
            all_predictions.append(sum(pred))
        # save prediction
        predictions = pd.DataFrame(zip(vpns, all_predictions), columns=["vpn", "predicted_qty"])
        predictions.to_csv(os.path.join(self.dir, "predictions.csv"), index=False)
