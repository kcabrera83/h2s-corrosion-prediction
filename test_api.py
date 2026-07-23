"""API integration tests for H2S corrosion prediction Flask app."""

import json
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:5010"

SAMPLE = {
    "h2s_concentration_ppm": 1000,
    "co2_concentration_pct": 5.0,
    "temperature_c": 90,
    "pressure_mpa": 10.0,
    "ph": 4.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel",
}


def test_health():
    r = requests.get(f"{BASE}/api/health")
    data = r.json()
    assert r.status_code == 200
    assert data["status"] == "healthy"
    assert data["predictor_loaded"] is True
    assert data["estimator_loaded"] is True
    print("[PASS] /api/health")


def test_models():
    r = requests.get(f"{BASE}/api/models")
    data = r.json()
    assert r.status_code == 200
    assert "models" in data
    assert len(data["models"]) >= 2
    print(f"[PASS] /api/models - {list(data['models'].keys())}")


def test_predict():
    r = requests.post(f"{BASE}/api/predict", json=SAMPLE)
    data = r.json()
    assert r.status_code == 200
    assert data["status"] == "ok"
    assert "corrosion_rate_mpy" in data
    assert isinstance(data["corrosion_rate_mpy"], (int, float))
    print(f"[PASS] /api/predict - corrosion_rate_mpy={data['corrosion_rate_mpy']}")


def test_life():
    r = requests.post(f"{BASE}/api/life", json=SAMPLE)
    data = r.json()
    assert r.status_code == 200
    assert data["status"] == "ok"
    assert "remaining_useful_life_years" in data
    assert isinstance(data["remaining_useful_life_years"], (int, float))
    print(f"[PASS] /api/life - remaining_useful_life_years={data['remaining_useful_life_years']}")


def test_various_materials():
    for mat in ["carbon_steel", "stainless_steel", "coated"]:
        payload = {**SAMPLE, "pipe_material": mat}
        r = requests.post(f"{BASE}/api/predict", json=payload)
        data = r.json()
        assert r.status_code == 200
        assert data["status"] == "ok"
        print(f"[PASS] predict with {mat} - rate={data['corrosion_rate_mpy']}")


def test_home():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200
    assert "H2S Corrosion" in r.text
    print("[PASS] /")


def main():
    print("=" * 50)
    print("  API Tests")
    print("=" * 50)
    try:
        test_health()
        test_models()
        test_predict()
        test_life()
        test_various_materials()
        test_home()
        print("\nAll tests passed!")
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Flask server at", BASE)
        print("Start the server first: python app.py")
        sys.exit(1)
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
