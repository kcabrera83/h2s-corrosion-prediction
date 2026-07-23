# User Guide - H2S Corrosion Prediction

## Overview

The H2S Corrosion Prediction System uses machine learning to predict hydrogen sulfide (H2S) induced corrosion rates and estimate remaining useful life for oil and gas pipelines.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd h2s-corrosion-prediction
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

This generates 3,000 synthetic corrosion samples and trains:
- Corrosion Rate Predictor (GradientBoostingRegressor)
- Remaining Useful Life Estimator (RandomForestRegressor)

### Run the Server

```bash
python app.py
```

Open `http://localhost:5010` in your browser.

## Dashboard Features

- **Corrosion Rate Panel** - Predict pipe corrosion rate (mpy) from environmental conditions
- **Remaining Life Panel** - Estimate pipeline remaining useful life (years)
- **Model Information** - View loaded models and their types
- **Dark Theme UI** - Modern dark-themed dashboard

## API Usage

### Predict Corrosion Rate (curl)

```bash
curl -X POST http://localhost:5010/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "h2s_concentration_ppm": 500,
    "co2_concentration_pct": 3.5,
    "temperature_c": 80,
    "pressure_mpa": 10.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel"
  }'
```

### Predict Corrosion Rate (Python)

```python
import requests

response = requests.post("http://localhost:5010/api/predict", json={
    "h2s_concentration_ppm": 500,
    "co2_concentration_pct": 3.5,
    "temperature_c": 80,
    "pressure_mpa": 10.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel"
})
result = response.json()
print(f"Corrosion rate: {result['corrosion_rate_mpy']} mpy")
```

### Estimate Remaining Life (curl)

```bash
curl -X POST http://localhost:5010/api/life \
  -H "Content-Type: application/json" \
  -d '{
    "h2s_concentration_ppm": 500,
    "co2_concentration_pct": 3.5,
    "temperature_c": 80,
    "pressure_mpa": 10.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel"
  }'
```

### Estimate Remaining Life (Python)

```python
import requests

response = requests.post("http://localhost:5010/api/life", json={
    "h2s_concentration_ppm": 500,
    "co2_concentration_pct": 3.5,
    "temperature_c": 80,
    "pressure_mpa": 10.0,
    "ph": 5.5,
    "flow_velocity_ms": 3.0,
    "pipe_material": "carbon_steel"
})
result = response.json()
print(f"Remaining life: {result['remaining_useful_life_years']} years")
```

### Check Health

```bash
curl http://localhost:5010/api/health
```

## Typical Workflow

1. Gather operating conditions (H2S/CO2 levels, temperature, pressure, pH, velocity, material)
2. Call `/api/predict` to get the corrosion rate
3. Call `/api/life` to estimate remaining useful life
4. Plan maintenance based on results

## Running Tests

```bash
python test_api.py
```

## Troubleshooting

- **Models not loaded**: Run `python train.py` first
- **Input error**: Ensure all 7 fields are provided with valid ranges
- **Pipe material**: Must be exactly carbon_steel, stainless_steel, or coated

---

*Elaborado por Ing. Kelvin Cabrera*
