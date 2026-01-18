# Deployment Guide

## Overview

This guide explains how to deploy the Day-0 Predictor for production use.

## Quick Start

### 1. Train Models

```bash
cd src
python train_and_deploy.py
```

This will:
- Train Day-0 predictor using only Day-0 available features
- Save models to `outputs/models/`
- Generate performance metrics

### 2. Start API Server

```bash
cd src
python deployment_api.py
```

The API will start on `http://localhost:5000`

### 3. Make Predictions

#### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "aadhaar_id": "test_123",
    "gender": "M",
    "state": "Karnataka",
    "district": "Bengaluru Urban",
    "urban_rural": "Urban",
    "eighteenth_birthday": "2028-05-15",
    "enrolment_date": "2010-05-15",
    "pincode": 560043
  }'
```

#### Batch Prediction

```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "citizens": [
      {
        "aadhaar_id": "test_1",
        "gender": "M",
        "state": "Karnataka",
        ...
      },
      ...
    ]
  }'
```

### 4. Command-Line Inference

```bash
# Single prediction
python inference.py --single --input input.json --output result.json

# Batch prediction from CSV
python inference.py --input batch_input.csv --output predictions.csv

# Interactive mode
python inference.py
```

## API Endpoints

### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true,
  "feature_engineer_loaded": true
}
```

### POST /predict
Predict failure risk for a single citizen.

**Request Body**:
```json
{
  "aadhaar_id": "sim_12345",
  "gender": "M",
  "state": "Karnataka",
  "district": "Bengaluru Urban",
  "urban_rural": "Urban",
  "eighteenth_birthday": "2028-05-15",
  "enrolment_date": "2010-05-15",
  "pincode": 560043
}
```

**Response**:
```json
{
  "aadhaar_id": "sim_12345",
  "prediction": 1,
  "predicted_class": "failure",
  "probability": 0.75,
  "risk_level": "high",
  "features_used": [...],
  "num_features": 13,
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /predict_batch
Predict for multiple citizens at once.

**Request Body**:
```json
{
  "citizens": [
    {...},
    {...}
  ]
}
```

**Response**:
```json
{
  "total": 10,
  "processed": 10,
  "failed": 0,
  "predictions": [...],
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /model_info
Get model information and metadata.

**Response**:
```json
{
  "model_type": "Day-0 Predictor",
  "purpose": "Predict transition failure risk when citizen turns 18",
  "features": [...],
  "num_features": 13,
  "models_available": ["xgboost"],
  "metadata": {...}
}
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY outputs/models/ ./models/

ENV PORT=5000
ENV HOST=0.0.0.0

CMD ["python", "src/deployment_api.py"]
```

Build and run:
```bash
docker build -t day0-predictor .
docker run -p 5000:5000 day0-predictor
```

### Environment Variables

- `PORT`: API port (default: 5000)
- `HOST`: API host (default: 0.0.0.0)
- `MODELS_DIR`: Path to models directory

### Requirements

Install dependencies:
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost
```

## Model Files

After training, the following files are created:

- `outputs/models/feature_engineer.pkl`: Feature engineering pipeline
- `outputs/models/day0_predictor_xgboost.pkl`: Day-0 predictor (XGBoost)
- `outputs/models/day0_predictor_random_forest.pkl`: Day-0 predictor (Random Forest)
- `outputs/models/day0_predictor_all.pkl`: All models (ensemble)

## Monitoring

### Health Checks

Monitor the `/health` endpoint to ensure the API is running and models are loaded.

### Logging

All API requests and predictions are logged. Check logs for:
- Prediction requests
- Errors
- Model loading status

## Performance

### Expected Latency

- Single prediction: < 100ms
- Batch prediction (100 records): < 5 seconds

### Throughput

- Can handle 100+ requests per second
- Batch processing optimized for bulk predictions

## Troubleshooting

### Model Not Loaded

If models are not loaded:
1. Ensure models are trained: `python train_and_deploy.py`
2. Check model files exist in `outputs/models/`
3. Check logs for loading errors

### Feature Engineering Errors

If feature engineering fails:
1. Ensure all required fields are provided
2. Check date formats (YYYY-MM-DD)
3. Verify categorical values match training data

### Low Prediction Accuracy

If predictions seem inaccurate:
1. Verify input data matches training distribution
2. Check feature engineering pipeline
3. Ensure models are using Day-0 features only

## Security Considerations

1. **Input Validation**: All inputs are validated
2. **Error Handling**: Errors don't expose model internals
3. **Rate Limiting**: Consider adding rate limiting for production
4. **Authentication**: Add authentication for production use
5. **HTTPS**: Use HTTPS in production

## Scaling

For high-volume deployments:
1. Use a WSGI server (gunicorn, uwsgi)
2. Load balance across multiple instances
3. Cache feature engineering results
4. Use async processing for batch predictions
