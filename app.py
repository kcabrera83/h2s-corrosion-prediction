import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

app = FastAPI(
    title="H2S Corrosion Prediction",
    description="Corrosion rate and remaining useful life prediction for H2S environments (scikit-survival + lifelines)",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = os.path.join("outputs", "models")

predictor_model = None
estimator_model = None
pipeline = None


def _build_pipeline():
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


@app.on_event("startup")
async def load_models():
    global predictor_model, estimator_model, pipeline
    try:
        for f in os.listdir(MODEL_DIR):
            if not f.endswith(".pkl"):
                continue
            path = os.path.join(MODEL_DIR, f)
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            if f == "pipeline.pkl":
                pipeline = obj
            elif "gbr" in f:
                from h2s_corrosion.models.corrosion_predictor import CorrosionPredictor
                predictor_model = CorrosionPredictor()
                predictor_model.model = obj
            elif "rfr" in f:
                from h2s_corrosion.models.life_estimator import LifeEstimator
                estimator_model = LifeEstimator()
                estimator_model.model = obj
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


class CorrosionRequest(BaseModel):
    h2s_concentration_ppm: float
    co2_concentration_pct: float
    temperature_c: float
    pressure_mpa: float
    ph: float
    flow_velocity_ms: float
    pipe_material: str


class CorrosionResponse(BaseModel):
    corrosion_rate_mpy: float
    status: str


class LifeResponse(BaseModel):
    remaining_useful_life_years: float
    status: str


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "predictor_loaded": predictor_model is not None,
        "estimator_loaded": estimator_model is not None,
        "framework": "scikit-survival/lifelines",
    }


@app.get("/api/models")
async def models_info():
    models = {}
    if predictor_model is not None:
        models["corrosion_rate_rsf"] = {
            "type": "RandomSurvivalForest (scikit-survival)",
            "n_estimators": getattr(predictor_model, "n_estimators", None),
            "framework": "scikit-survival",
        }
    if estimator_model is not None:
        models["life_estimator_weibull"] = {
            "type": "WeibullFitter (lifelines)",
            "framework": "lifelines",
        }
    return {"models": models, "status": "ok"}


@app.post("/api/predict", response_model=CorrosionResponse)
async def predict(request: CorrosionRequest):
    try:
        X = _transform_input(request.model_dump())
        median_times = predictor_model.predict_median_survival(X)
        rate = float(median_times[0]) if median_times else 0.0
        return CorrosionResponse(corrosion_rate_mpy=round(rate, 2), status="ok")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/life", response_model=LifeResponse)
async def life(request: CorrosionRequest):
    try:
        X = _transform_input(request.model_dump())
        life_years = float(estimator_model.predict(X)[0])
        return LifeResponse(remaining_useful_life_years=round(life_years, 2), status="ok")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
