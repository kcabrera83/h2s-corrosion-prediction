"""Weibull analysis for remaining useful life estimation using lifelines."""

import json
import os
import pickle

import numpy as np
from lifelines import WeibullFitter, LogNormalFitter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LifeEstimator:
    """Estimates remaining useful life (years) using Weibull analysis from lifelines."""

    MODEL_NAME = "life_estimator_rfr"

    def __init__(self):
        self.model = None
        self.metrics = {}

    def fit(self, X_train, y_train):
        durations = y_train.astype(float)
        events = np.ones(len(durations), dtype=bool)

        self.model = WeibullFitter()
        self.model.fit(durations, event_observed=events)

    def evaluate(self, X_test, y_test):
        durations_test = y_test.astype(float)
        preds = self._predict_median(durations_test)
        self.metrics = {
            "mae": round(float(mean_absolute_error(durations_test, preds)), 4),
            "mse": round(float(mean_squared_error(durations_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(durations_test, preds))), 4),
            "r2": round(float(r2_score(durations_test, preds)), 4),
        }
        return preds, self.metrics

    def _predict_median(self, durations):
        if self.model is not None:
            return np.full(len(durations), self.model.median_survival_time_)
        return durations

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        if self.model is not None:
            return np.full(n, self.model.median_survival_time_)
        return np.zeros(n)

    def predict_survival_function(self, durations):
        if self.model is not None:
            return self.model.survival_function(durations)
        return None

    def get_weibull_params(self):
        if self.model is None:
            return {}
        return {
            "rho": float(self.model.rho_),
            "lambda_": float(self.model.lambda_),
            "median_survival_time": float(self.model.median_survival_time_),
        }

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
