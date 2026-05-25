# Project: Sentinel - Server Anomaly Detection API

## 1. Objective
Build a lightweight, production-ready Machine Learning API system that detects anomalies in server metrics. The system uses the Pooled Server Metrics (PSM) dataset (real-world telemetry data from eBay) to train an unsupervised anomaly detection model. The application must be fully tested, containerized, and optimized for low-latency inference.

## 2. Tech Stack
* **Language:** Python 3.10+
* **Data Processing & ML:** scikit-learn, pandas, numpy
* **Model:** Isolation Forest (Unsupervised Anomaly Detection)
* **Backend/API:** FastAPI, Uvicorn, Pydantic
* **Testing:** Pytest, httpx
* **Deployment:** Docker

## 3. System Architecture & Flow
1. **Model Training (Offline):** - A standalone script (`src/train.py`) reads the `server_metrics.csv` (PSM dataset).
   - It drops non-numeric columns (like timestamps) and selects key telemetry features.
   - It scales the data using `StandardScaler` and trains an `IsolationForest` model.
   - The artifacts are exported as `isolation_forest.pkl` and `scaler.pkl` into the `models/` directory.
2. **Inference API (Online):** - A FastAPI server (`src/main.py`) loads the `.pkl` artifacts into memory on startup.
   - It exposes a `POST /api/v1/analyze` endpoint accepting a JSON payload of server metrics.
3. **Response:** - The API scales the incoming payload, predicts the state, and returns a JSON response classifying the server state as "Normal" or "Anomaly".

## 4. API Contract
**Endpoint:** `POST /api/v1/analyze`

**Request Payload Example (JSON):**
*(Note: Feature names will dynamically map to the top 5 numeric features extracted from the PSM dataset. Below is a structural example)*
{
  "feature_1": 0.85,
  "feature_2": 12.4,
  "feature_3": 512.0,
  "feature_4": 0.05,
  "feature_5": 1024.5
}

**Response Payload (JSON):**
{
  "status": "success",
  "prediction": "Anomaly",
  "timestamp": "2026-05-25T22:30:00Z"
}

## 5. Expected Directory Structure
/sentinel
  ├── data/                  
  │   └── server_metrics.csv # The extracted PSM dataset (eBay telemetry)
  ├── models/                # Stores isolation_forest.pkl and scaler.pkl
  ├── src/
  │   ├── train.py           # ML training script (Data prep + Isolation Forest)
  │   ├── main.py            # FastAPI application setup & endpoints
  │   └── schemas.py         # Pydantic models for API request/response validation
  ├── tests/
  │   └── test_api.py        # Pytest cases for FastAPI endpoints (valid, invalid, missing fields)
  ├── requirements.txt       # Dependencies
  ├── .dockerignore
  └── Dockerfile             # Multi-stage or optimized Dockerfile using python:3.10-slim