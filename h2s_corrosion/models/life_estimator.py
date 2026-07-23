"""RandomForest model for remaining useful life estimation."""

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LifeEstimator:
    """Estimates remaining useful life (years) using RandomForest."""

    MODEL_NAME = "life_estimator_rfr"

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
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
