"""Flask API for H2S corrosion prediction."""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join("outputs", "models")

predictor_model = None
estimator_model = None
pipeline = None


def _load_models():
    global predictor_model, estimator_model, pipeline
    for f in os.listdir(MODEL_DIR):
        if not f.endswith(".pkl"):
            continue
        path = os.path.join(MODEL_DIR, f)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if f == "pipeline.pkl":
            pipeline = obj
        elif "gbr" in f:
            predictor_model = obj
        elif "rfr" in f:
            estimator_model = obj


def _build_pipeline():
    """Rebuild pipeline from training config."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    NUMERIC = [
        "h2s_concentration_ppm", "co2_concentration_pct",
        "temperature_c", "pressure_mpa", "ph", "flow_velocity_ms",
    ]
    CATEG = ["pipe_material"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC),
            ("cat", categorical_transformer, CATEG),
        ],
        remainder="drop",
    )


def _transform_input(data):
    """Transform a single sample dict into model-ready array."""
    import pandas as pd

    NUMERIC = [
        "h2s_concentration_ppm", "co2_concentration_pct",
        "temperature_c", "pressure_mpa", "ph", "flow_velocity_ms",
    ]
    row = [data[k] for k in NUMERIC]
    material = data["pipe_material"]

    df_num = pd.DataFrame([row], columns=NUMERIC)
    df_cat = pd.DataFrame([[material]], columns=["pipe_material"])
    df = pd.concat([df_num, df_cat], axis=1)

    if pipeline is not None:
        return pipeline.transform(df)

    pipe = _build_pipeline()
    return pipe.fit_transform(df)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        X = _transform_input(data)
        rate = float(predictor_model.predict(X)[0])
        return jsonify({"corrosion_rate_mpy": round(rate, 2), "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


@app.route("/api/life", methods=["POST"])
def api_life():
    try:
        data = request.get_json(force=True)
        X = _transform_input(data)
        life = float(estimator_model.predict(X)[0])
        return jsonify({"remaining_useful_life_years": round(life, 2), "status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 400


@app.route("/api/models", methods=["GET"])
def api_models():
    models = {}
    if predictor_model is not None:
        models["corrosion_rate_gbr"] = {
            "type": type(predictor_model).__name__,
            "n_estimators": getattr(predictor_model, "n_estimators", None),
        }
    if estimator_model is not None:
        models["life_estimator_rfr"] = {
            "type": type(estimator_model).__name__,
            "n_estimators": getattr(estimator_model, "n_estimators", None),
        }
    return jsonify({"models": models, "status": "ok"})


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "healthy",
        "predictor_loaded": predictor_model is not None,
        "estimator_loaded": estimator_model is not None,
    })


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "H2S Corrosion Prediction", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/predict": {"post": {"summary": "Predict corrosion rate in mpy"}},
            "/api/life": {"post": {"summary": "Estimate remaining useful life in years"}},
        }
    })


if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5010, debug=True)
