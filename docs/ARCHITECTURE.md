# Architecture - H2S Corrosion Prediction

## System Overview

```
+------------------+     +-------------------+     +------------------+
|   Data Layer     | --> |   Model Layer     | --> |   API Layer      |
| (Data Generator) |     | (ML Models)       |     | (Flask REST)     |
+------------------+     +-------------------+     +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | Dashboard Layer  |
                                                 | (HTML/CSS/JS)    |
                                                 +------------------+
```

## Components

### Data Layer

- **Source**: Synthetic data generator (`CorrosionDataGenerator`)
- **Samples**: 3,000 corrosion records
- **Parameters**: H2S concentration, CO2 concentration, temperature, pressure, pH, flow velocity, pipe material
- **Targets**: corrosion_rate_mpy (regression), remaining_useful_life_years (regression)

### Model Layer

#### Corrosion Rate Predictor
- **Algorithm**: GradientBoostingRegressor
- **Task**: Predict corrosion rate in mils per year (mpy)
- **Input**: 7 features (6 numeric + 1 categorical)
- **Output**: Corrosion rate (mpy)
- **Serialization**: pickle (`.pkl`)

#### Remaining Useful Life Estimator
- **Algorithm**: RandomForestRegressor
- **Task**: Predict remaining useful life in years
- **Input**: Same 7 features as predictor
- **Output**: Remaining life (years)
- **Serialization**: pickle (`.pkl`)

### Preprocessing Pipeline

- **ColumnTransformer** with:
  - StandardScaler for 6 numeric features
  - OneHotEncoder for pipe_material (3 categories)
- Pipeline object saved as `pipeline.pkl` for consistent transforms at inference

### API Layer

- **Framework**: Flask
- **Port**: 5010
- **Format**: JSON request/response
- **Endpoints**: 5 (predict, life, health, models, docs)

### Dashboard Layer

- **Frontend**: HTML/CSS/JS (Jinja2 templates)
- **Theme**: Dark theme UI

## Data Flow

1. **Input** -> JSON with 7 corrosion parameters
2. **Transform** -> ColumnTransformer applies scaling + encoding
3. **Predict** -> GradientBoosting (rate) or RandomForest (life)
4. **Output** -> JSON with prediction and status

## Feature Processing

| Feature | Transform | Description |
|---------|-----------|-------------|
| h2s_concentration_ppm | StandardScaler | H2S concentration |
| co2_concentration_pct | StandardScaler | CO2 concentration |
| temperature_c | StandardScaler | Temperature |
| pressure_mpa | StandardScaler | Pressure |
| ph | StandardScaler | Fluid pH |
| flow_velocity_ms | StandardScaler | Flow velocity |
| pipe_material | OneHotEncoder | Pipe material (3 categories) |

## Project Structure

```
h2s-corrosion-prediction/
├── h2s_corrosion/
│   ├── __init__.py
│   ├── data_generator.py          # Synthetic data generation
│   ├── models/
│   │   ├── corrosion_predictor.py # GradientBoosting regressor
│   │   └── life_estimator.py      # RandomForest regressor
│   └── utils/
│       └── preprocessor.py        # ColumnTransformer pipeline
├── outputs/models/                # Saved model artifacts
├── templates/
│   └── index.html                 # Dashboard UI
├── train.py                       # Training pipeline
├── app.py                         # Flask API server
├── test_api.py                    # API test suite
├── requirements.txt
└── setup.py
```

## Model Evaluation

### Corrosion Rate Predictor
- RMSE: evaluated on test set
- R2 score: evaluated on test set

### Remaining Useful Life Estimator
- RMSE: evaluated on test set
- R2 score: evaluated on test set

---

*Elaborado por Ing. Kelvin Cabrera*
