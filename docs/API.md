# API Documentation - H2S Corrosion Prediction

## Base URL

```
http://localhost:5010
```

## Endpoints

### GET /

Serve the main web dashboard UI.

**Response:** HTML page

---

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "predictor_loaded": true,
  "estimator_loaded": true
}
```

---

### GET /api/models

Return information about loaded models.

**Response:**
```json
{
  "models": {
    "corrosion_rate_gbr": {
      "type": "GradientBoostingRegressor",
      "n_estimators": 200
    },
    "life_estimator_rfr": {
      "type": "RandomForestRegressor",
      "n_estimators": 200
    }
  },
  "status": "ok"
}
```

---

### POST /api/predict

Predict corrosion rate in mils per year (mpy).

**Request:**
```json
{
  "h2s_concentration_ppm": 500,
  "co2_concentration_pct": 3.5,
  "temperature_c": 80,
  "pressure_mpa": 10.0,
  "ph": 5.5,
  "flow_velocity_ms": 3.0,
  "pipe_material": "carbon_steel"
}
```

**Required Fields:**

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| h2s_concentration_ppm | float | H2S gas concentration | 50 - 5000 ppm |
| co2_concentration_pct | float | CO2 gas concentration | 0.1 - 15 % |
| temperature_c | float | Operating temperature | 20 - 200 C |
| pressure_mpa | float | Operating pressure | 0.1 - 25 MPa |
| ph | float | Fluid pH level | 2.5 - 8.5 |
| flow_velocity_ms | float | Flow velocity | 0.1 - 15 m/s |
| pipe_material | string | Pipe material type | carbon_steel / stainless_steel / coated |

**Response:**
```json
{
  "corrosion_rate_mpy": 12.45,
  "status": "ok"
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Invalid input | `{"error": "<details>", "status": "error"}` |

---

### POST /api/life

Estimate remaining useful life in years.

**Request:**
```json
{
  "h2s_concentration_ppm": 500,
  "co2_concentration_pct": 3.5,
  "temperature_c": 80,
  "pressure_mpa": 10.0,
  "ph": 5.5,
  "flow_velocity_ms": 3.0,
  "pipe_material": "carbon_steel"
}
```

**Response:**
```json
{
  "remaining_useful_life_years": 8.23,
  "status": "ok"
}
```

**Error Responses:**
| Status | Condition | Body |
|--------|-----------|------|
| 400 | Invalid input | `{"error": "<details>", "status": "error"}` |

---

### GET /api/docs

Return OpenAPI 3.0 specification.

---

## Input Parameters Reference

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| h2s_concentration_ppm | ppm | 50 - 5000 | H2S gas concentration in the fluid |
| co2_concentration_pct | % | 0.1 - 15 | CO2 gas concentration |
| temperature_c | C | 20 - 200 | Operating temperature |
| pressure_mpa | MPa | 0.1 - 25 | Operating pressure |
| pH | - | 2.5 - 8.5 | Fluid acidity |
| flow_velocity_ms | m/s | 0.1 - 15 | Fluid flow velocity |
| pipe_material | - | carbon_steel, stainless_steel, coated | Pipe material type |

## Error Codes

- **200**: Success
- **400**: Bad request (invalid input)
- **500**: Internal server error

---

*Elaborado por Ing. Kelvin Cabrera*
