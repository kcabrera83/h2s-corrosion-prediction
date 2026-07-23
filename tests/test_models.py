import pytest
import os
import pickle
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_predictor_model_loads():
    path = os.path.join(MODELS_DIR, "corrosion_rate_gbr.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_estimator_model_loads():
    path = os.path.join(MODELS_DIR, "life_estimator_rfr.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_pipeline_loads():
    path = os.path.join(MODELS_DIR, "pipeline.pkl")
    assert os.path.exists(path)
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    assert pipeline is not None


def test_corrosion_prediction():
    with open(os.path.join(MODELS_DIR, "corrosion_rate_gbr.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)

    df = pd.DataFrame([{
        "h2s_concentration_ppm": 500.0,
        "co2_concentration_pct": 3.0,
        "temperature_c": 80.0,
        "pressure_mpa": 5.0,
        "ph": 5.5,
        "flow_velocity_ms": 3.0,
        "pipe_material": "carbon_steel",
    }])
    X = pipeline.transform(df)
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0


def test_life_prediction():
    with open(os.path.join(MODELS_DIR, "life_estimator_rfr.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "pipeline.pkl"), "rb") as f:
        pipeline = pickle.load(f)

    df = pd.DataFrame([{
        "h2s_concentration_ppm": 500.0,
        "co2_concentration_pct": 3.0,
        "temperature_c": 80.0,
        "pressure_mpa": 5.0,
        "ph": 5.5,
        "flow_velocity_ms": 3.0,
        "pipe_material": "carbon_steel",
    }])
    X = pipeline.transform(df)
    pred = model.predict(X)
    assert pred is not None
    assert len(pred) == 1
    assert pred[0] > 0
