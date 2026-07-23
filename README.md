# H2S Corrosion Prediction System

Machine learning-based prediction system for hydrogen sulfide (H2S) induced corrosion in oil and gas pipelines.

## Overview

This project uses ensemble ML models to predict:
- **Corrosion rate** (mpy) using Gradient Boosting Regression
- **Remaining useful life** (years) using Random Forest Regression

## Features

- Synthetic data generation for H2S/CO2 corrosion scenarios
- Preprocessing pipeline with scaling and one-hot encoding
- REST API for real-time predictions
- Web dashboard with dark theme UI
- CI/CD pipeline via GitHub Actions

## Installation

```bash
git clone https://github.com/user/h2s-corrosion-prediction.git
cd h2s-corrosion-prediction
pip install -r requirements.txt
```

## Usage

Train the models:
```bash
python train.py
```

Start the API server:
```bash
python app.py
```

Run API tests:
```bash
python test_api.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web dashboard |
| POST | `/api/predict` | Predict corrosion rate |
| POST | `/api/life` | Predict remaining useful life |
| GET | `/api/models` | List loaded models |
| GET | `/api/health` | Health check |

## Input Parameters

| Parameter | Range | Unit |
|-----------|-------|------|
| H2S concentration | 50 - 5000 | ppm |
| CO2 concentration | 0.1 - 15 | % |
| Temperature | 20 - 200 | C |
| Pressure | 0.1 - 25 | MPa |
| pH | 2.5 - 8.5 | - |
| Flow velocity | 0.1 - 15 | m/s |
| Pipe material | carbon_steel / stainless_steel / coated | - |

## Project Structure

```
h2s-corrosion-prediction/
├── h2s_corrosion/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── models/
│   │   ├── corrosion_predictor.py
│   │   └── life_estimator.py
│   └── utils/
│       └── preprocessor.py
├── outputs/models/
├── templates/
│   └── index.html
├── train.py
├── app.py
├── test_api.py
├── requirements.txt
└── setup.py
```

## Elaborado por Ing. Kelvin Cabrera
