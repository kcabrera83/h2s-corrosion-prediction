import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app, _load_models

_load_models()
client = app.test_client()

SAMPLE_INPUT = {
    "h2s_concentration_ppm": 500.0,
    "co2_concentration_pct": 3.0,
    "temperature_c": 80.0,
    "pressure_mpa": 5.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel",
}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["predictor_loaded"] is True
    assert data["estimator_loaded"] is True


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    assert "corrosion_rate_gbr" in data["models"]
    assert "life_estimator_rfr" in data["models"]


def test_api_docs():
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"


def test_predict_valid():
    response = client.post("/api/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "corrosion_rate_mpy" in data
    assert data["status"] == "ok"
    assert data["corrosion_rate_mpy"] > 0


def test_life_valid():
    response = client.post("/api/life", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.get_json()
    assert "remaining_useful_life_years" in data
    assert data["status"] == "ok"
    assert data["remaining_useful_life_years"] > 0


def test_predict_missing_fields():
    response = client.post("/api/predict", json={})
    assert response.status_code == 400


def test_life_missing_fields():
    response = client.post("/api/life", json={})
    assert response.status_code == 400


def test_predict_all_materials():
    for material in ["carbon_steel", "stainless_steel", "coated"]:
        response = client.post("/api/predict", json={
            **SAMPLE_INPUT,
            "pipe_material": material,
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["corrosion_rate_mpy"] > 0
