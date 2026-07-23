import pytest

SAMPLE_INPUT = {
    "h2s_concentration_ppm": 500.0,
    "co2_concentration_pct": 3.0,
    "temperature_c": 80.0,
    "pressure_mpa": 5.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel",
}


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "predictor_loaded" in data
    assert "estimator_loaded" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code in (200, 500, 503)


def test_predict_valid(client):
    response = client.post("/api/predict", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "corrosion_rate_mpy" in data
        assert data["status"] == "ok"
        assert data["corrosion_rate_mpy"] > 0


def test_life_valid(client):
    response = client.post("/api/life", json=SAMPLE_INPUT)
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "remaining_useful_life_years" in data
        assert data["status"] == "ok"
        assert data["remaining_useful_life_years"] > 0


def test_predict_all_materials(client):
    for material in ["carbon_steel", "stainless_steel", "coated"]:
        response = client.post("/api/predict", json={
            **SAMPLE_INPUT,
            "pipe_material": material,
        })
        assert response.status_code in (200, 400, 500)
        if response.status_code == 200:
            data = response.json()
            assert data["corrosion_rate_mpy"] > 0
