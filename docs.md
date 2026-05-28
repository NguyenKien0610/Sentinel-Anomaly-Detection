# Project: Sentinel v2.0 - Advanced Server Anomaly Detection System

## 1. Objective
Upgrade the existing FastAPI anomaly detection service into a production-grade, observable system. The system must now persist prediction logs to PostgreSQL asynchronously and expose real-time metrics for Prometheus and Grafana.

## 2. Tech Stack
* **ML Model:** scikit-learn (Isolation Forest)
* **Backend:** FastAPI, SQLAlchemy (Async)
* **Database:** PostgreSQL
* **Observability:** prometheus_client, Prometheus, Grafana
* **Containerization:** Docker Compose

## 3. Advanced System Architecture
1. **Inference & Metrics:** When `POST /api/v1/analyze` is called, the model predicts the anomaly. The API increments Prometheus counters (e.g., `total_requests`, `anomaly_detected_total`).
2. **Async Persistence:** The API uses FastAPI's `BackgroundTasks` (or async SQLAlchemy) to save the request payload and prediction result into a PostgreSQL table named `prediction_logs` without blocking the HTTP response.
3. **Metrics Endpoint:** The API exposes a `GET /metrics` endpoint for Prometheus to scrape.
4. **Orchestration:** A `docker-compose.yml` file orchestrates 4 services: `api`, `postgres`, `prometheus`, and `grafana`.

## 4. Database Schema (PostgreSQL)
Table: `prediction_logs`
- `id`: Integer, Primary Key
- `feature_15`, `feature_16`, `feature_19`, `feature_20`, `feature_9`: Float
- `prediction`: String ("Normal" or "Anomaly")
- `timestamp`: DateTime

## 5. Directory Structure Updates
/sentinel
  ├── src/
  │   ├── database.py        # SQLAlchemy async setup and session maker
  │   ├── models_db.py       # SQLAlchemy ORM models (PredictionLog)
  │   ├── main.py            # Updated to include Prometheus metrics and DB injection
  ...
  ├── docker-compose.yml     # Orchestrates api, db, prometheus, grafana
  └── prometheus.yml         # Prometheus configuration file