# Deployment Guide - H2S Corrosion Prediction

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

EXPOSE 5010

CMD ["python", "app.py"]
```

### Build and Run

```bash
docker build -t h2s-corrosion-prediction .
docker run -p 5010:5010 h2s-corrosion-prediction
```

## Docker Compose

```yaml
version: '3.8'
services:
  h2s-corrosion-prediction:
    build: .
    ports:
      - "5010:5010"
    environment:
      - FLASK_ENV=production
    volumes:
      - model-data:/app/outputs

volumes:
  model-data:
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PORT | Server port | 5010 |

## Production Considerations

- Use gunicorn for production serving:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5010 app:app
  ```
- Set `debug=False` in `app.py` (set to `True` by default - change for production)
- Configure reverse proxy (nginx) for SSL termination
- Set up health check monitoring on `/api/health`
- Use a process manager (systemd, supervisor) for auto-restart

## Training Pipeline

1. `python train.py` generates synthetic data and trains models
2. Models + preprocessing pipeline saved to `outputs/models/`
3. `python app.py` loads models and starts the API server

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push to main
- Installs dependencies
- Runs training pipeline
- Executes API tests

---

*Elaborado por Ing. Kelvin Cabrera*
