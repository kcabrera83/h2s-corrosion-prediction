import json
import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class CorrosionPredictor:
    """Predicts corrosion rate (mpy) using Random Survival Forest from scikit-survival."""

    MODEL_NAME = "corrosion_rate_gbr"

    def __init__(self):
        self.model = None
        self.metrics = {}

    def fit(self, X_train, y_train):
        from sksurv.ensemble import RandomSurvivalForest
        durations = y_train.astype(float)
        events = np.ones(len(durations), dtype=bool)

        self.model = RandomSurvivalForest(
            n_estimators=200,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=2024,
        )

        y_struct = np.array(
            [(bool(e), float(d)) for e, d in zip(events, durations)],
            dtype=[("event", bool), ("time", float)],
        )
        self.model.fit(X_train, y_struct)

    def evaluate(self, X_test, y_test):
        durations_test = y_test.astype(float)
        events_test = np.ones(len(durations_test), dtype=bool)
        y_struct_test = np.array(
            [(bool(e), float(d)) for e, d in zip(events_test, durations_test)],
            dtype=[("event", bool), ("time", float)],
        )
        preds = self.model.predict(X_test)
        self.metrics = {
            "mae": round(float(mean_absolute_error(durations_test, preds)), 4),
            "mse": round(float(mean_squared_error(durations_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(durations_test, preds))), 4),
            "r2": round(float(r2_score(durations_test, preds)), 4),
        }
        return preds, self.metrics

    def predict(self, X):
        return self.model.predict(X)

    def predict_median_survival(self, X):
        surv_funcs = self.model.predict_survival_function(X)
        median_times = []
        for surv in surv_funcs:
            times = surv.x
            values = surv(times)
            idx = np.searchsorted(values, 0.5, side="right")
            if idx < len(times):
                median_times.append(float(times[idx]))
            else:
                median_times.append(float(times[-1]))
        return median_times

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
