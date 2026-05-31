# Sentinel

Sentinel is a FastAPI-based anomaly detection service for server telemetry. It uses an Isolation Forest model trained on the Pooled Server Metrics (PSM) dataset, exposes an inference API, persists prediction logs to PostgreSQL asynchronously, and publishes Prometheus metrics for observability, input drift monitoring, and model lifecycle tracking.

The current project includes:

- Offline training with `scikit-learn`
- Real-time inference with FastAPI
- A built-in web UI at `/`
- Async PostgreSQL logging with SQLAlchemy
- Alembic database migrations
- Prometheus metrics at `/metrics`
- Drift monitoring against training baselines
- Local model registry with active-version manifest
- Retraining metadata for reproducibility and rollback
- Health and readiness probes
- Structured JSON logging with request tracing
- Pre-provisioned Grafana dashboard for ML monitoring
- Docker Compose orchestration for API, PostgreSQL, Prometheus, and Grafana

## Architecture

1. `src/train.py` reads `data/server_metrics.csv`, removes timestamp columns, keeps numeric columns, selects 5 features, scales them with `StandardScaler`, trains `IsolationForest`, creates a versioned model bundle under `models/registry/`, and updates `models/current_model.json`.
2. Each model bundle stores the model, scaler, monitoring baseline, and training metadata such as dataset hash, selected features, model parameters, anomaly ratio, sample payload, and training library versions.
3. `src/main.py` wires the application layers together, resolves the active model from the registry manifest, serves the frontend, and exposes API endpoints.
4. `POST /api/v1/analyze` validates the payload, scales the input, runs anomaly detection, increments Prometheus counters, and writes the request plus prediction into PostgreSQL in a background task.
5. Alembic manages schema evolution for PostgreSQL, and Docker Compose runs `alembic upgrade head` before starting the API container.
6. Prometheus scrapes `GET /metrics`, while Grafana auto-loads a dashboard for request volume, anomaly rate, latency, drift metrics, and active model behavior.

## Tech Stack

- Python 3.10+
- FastAPI
- scikit-learn (pinned)
- pandas (pinned)
- numpy (pinned)
- SQLAlchemy async
- asyncpg
- PostgreSQL
- Prometheus
- Grafana
- Docker / Docker Compose

## Project Layout

```text
sentinel/
|-- data/
|   `-- server_metrics.csv
|-- models/
|   |-- current_model.json
|   |-- isolation_forest.pkl
|   |-- monitoring_baseline.json
|   |-- registry/
|   `-- scaler.pkl
|-- grafana/
|   |-- dashboards/
|   `-- provisioning/
|-- alembic/
|   `-- versions/
|-- src/
|   |-- api/
|   |-- core/
|   |-- db/
|   |-- services/
|   |-- database.py
|   |-- main.py
|   |-- models_db.py
|   |-- schemas.py
|   |-- train.py
|   `-- static/
|       `-- index.html
|-- tests/
|   `-- test_api.py
|-- docker-compose.yml
|-- Dockerfile
|-- prometheus.yml
`-- requirements.txt
```

## API Contract

### `POST /api/v1/analyze`

Request body:

```json
{
  "feature_15": 0.3187968986906816,
  "feature_16": 0.4518560659934575,
  "feature_19": 0.6098826098826098,
  "feature_20": 0.0084317032040472,
  "feature_9": 0.4470765464645514
}
```

Response body:

```json
{
  "status": "success",
  "prediction": "Anomaly",
  "timestamp": "2026-05-28T08:16:00.495383Z"
}
```

Other useful endpoints:

- `GET /` serves the frontend UI
- `GET /docs` serves Swagger UI
- `GET /healthz` returns liveness status
- `GET /readyz` returns readiness checks for model, drift baseline, and database
- `GET /metrics` exposes Prometheus metrics
- `GET /api/v1/monitoring/drift` returns current drift-monitoring status
- `GET /api/v1/model/info` returns the active model version and training metadata

## Database Schema

Table: `prediction_logs`

- `id` - integer primary key
- `feature_15` - float
- `feature_16` - float
- `feature_19` - float
- `feature_20` - float
- `feature_9` - float
- `prediction` - string (`Normal` or `Anomaly`)
- `timestamp` - datetime

The schema is managed by Alembic. The initial migration creates `prediction_logs` and records the revision in `alembic_version`.

## Local Run

### Option 1: Docker Compose

This is the default way to run the full stack.

```bash
docker compose up --build
```

Available services:

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000/`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)
- PostgreSQL: `localhost:5432`

The API container receives:

```text
DATABASE_URL=postgresql+asyncpg://sentinel:sentinel_password@db:5432/sentinel_db
```

The API service runs migrations automatically on startup:

```bash
alembic upgrade head
```

### Option 2: Local Python Environment

You need PostgreSQL running separately and must provide `DATABASE_URL`.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:DATABASE_URL="postgresql+asyncpg://sentinel:sentinel_password@localhost:5432/sentinel_db"
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="postgresql+asyncpg://sentinel:sentinel_password@localhost:5432/sentinel_db"
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## Model Training

To retrain the model from `data/server_metrics.csv`:

```bash
python src/train.py
```

The script updates:

- `models/current_model.json`
- `models/isolation_forest.pkl`
- `models/scaler.pkl`
- `models/monitoring_baseline.json`
- `models/registry/<model_version>/model.pkl`
- `models/registry/<model_version>/scaler.pkl`
- `models/registry/<model_version>/monitoring_baseline.json`
- `models/registry/<model_version>/metadata.json`

It also prints the new `model_version` plus a sample JSON payload for API testing.

The direct dependencies are pinned in [requirements.txt](E:/Project/sentinel/requirements.txt) so the training path and Docker runtime stay aligned for the ML stack.

The active manifest looks like this at a high level:

```json
{
  "active_version": "20260531T150617Z-816e0e2a",
  "metadata_path": "models/registry/20260531T150617Z-816e0e2a/metadata.json",
  "artifacts": {
    "model": "models/registry/20260531T150617Z-816e0e2a/model.pkl",
    "scaler": "models/registry/20260531T150617Z-816e0e2a/scaler.pkl",
    "monitoring_baseline": "models/registry/20260531T150617Z-816e0e2a/monitoring_baseline.json"
  }
}
```

## Testing

Run the API tests with:

```bash
py -m pytest -q
```

Current tests cover:

- Valid prediction request returns `200`
- Missing required fields returns `422`
- Invalid field types returns `422`

## Observability

Prometheus scrapes the API every 5 seconds using `prometheus.yml`.

Current monitoring metrics include:

- `total_requests_total`
- `anomaly_detected_total`
- `prediction_label_total`
- `request_latency_seconds`
- `input_drift_score`
- `drift_alert_total`
- `feature_abs_zscore`
- `feature_input_value`

## Logging And Tracing

The API emits structured JSON logs to stdout. Each request gets an `X-Request-ID` header, and the same request id is attached to access logs and inference-related application logs.

This makes it easier to:

- trace a single request across logs
- correlate client-side errors with backend events
- ship logs into systems such as Loki, Elasticsearch, or Datadog later

## Notes

- `src/main.py` currently uses FastAPI `@app.on_event("startup")`. It works, but FastAPI now recommends lifespan handlers for new code.
- The project now uses a cleaner backend split across `core`, `db`, `services`, and `api`, while keeping `src.main:app` as the runtime entrypoint.

## License

This project is licensed under the MIT License.
