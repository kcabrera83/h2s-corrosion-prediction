"""GradientBoosting model for corrosion rate prediction."""

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class CorrosionPredictor:
    """Predicts corrosion rate (mpy) using GradientBoosting."""

    MODEL_NAME = "corrosion_rate_gbr"

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self.metrics = {}

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        self.metrics = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "mse": round(float(mean_squared_error(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
        }
        return preds, self.metrics

    def predict(self, X):
        return self.model.predict(X)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{self.MODEL_NAME}.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(directory, f"{self.MODEL_NAME}_metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)

    @classmethod
    def load(cls, directory):
        obj = cls()
        with open(os.path.join(directory, f"{obj.MODEL_NAME}.pkl"), "rb") as f:
            obj.model = pickle.load(f)
        metrics_path = os.path.join(directory, f"{obj.MODEL_NAME}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                obj.metrics = json.load(f)
        return obj
