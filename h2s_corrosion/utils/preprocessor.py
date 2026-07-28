import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "h2s_concentration_ppm",
    "co2_concentration_pct",
    "temperature_c",
    "pressure_mpa",
    "ph",
    "flow_velocity_ms",
]

CATEGORICAL_FEATURES = ["pipe_material"]

TARGET_CORROSION = "corrosion_rate_mpy"
TARGET_LIFE = "remaining_useful_life_years"

PIPELINE_VERSION = "1.0"


class CorrosionPreprocessor:
    """Builds sklearn preprocessing pipelines and prepares train/test splits."""

    def __init__(self):
        self._pipeline = None
        self._feature_names = None

    @property
    def feature_names(self):
        return self._feature_names

    def _build_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler()),
        ])

        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )

    def _derive_life(self, df):
        """Derive remaining useful life from corrosion rate."""
        wall_thickness = 12.0
        remaining = np.maximum(wall_thickness - df["wall_loss_mm"], 0.1)
        df[TARGET_LIFE] = np.round(remaining / (df[TARGET_CORROSION] * 0.001 + 1e-6), 2)
        df[TARGET_LIFE] = np.clip(df[TARGET_LIFE], 0.1, 50.0)
        return df

    def prepare(self, df, target=TARGET_CORROSION, test_size=0.2, random_state=2024):
        """Return X_train, X_test, y_train, y_test, fitted pipeline."""
        df = self._derive_life(df.copy())

        self._pipeline = self._build_pipeline()
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_t = self._pipeline.fit_transform(X_train)
        X_test_t = self._pipeline.transform(X_test)

        cat_names = list(
            self._pipeline
            .named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
        self._feature_names = NUMERIC_FEATURES + cat_names

        return X_train_t, X_test_t, y_train.values, y_test.values, self._pipeline

    def transform(self, df):
        """Transform new data using fitted pipeline."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call prepare() first.")
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        return self._pipeline.transform(X)
