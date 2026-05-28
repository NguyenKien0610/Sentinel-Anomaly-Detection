import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

try:
    from src.schemas import ServerMetrics
except ModuleNotFoundError:
    from schemas import ServerMetrics

DB_LOGGING_AVAILABLE = True
DB_LOGGING_ERROR: str | None = None

try:
    from src.database import AsyncSessionLocal, init_db
    from src.models_db import PredictionLog
except Exception as exc:  # noqa: BLE001
    DB_LOGGING_AVAILABLE = False
    DB_LOGGING_ERROR = str(exc)


app = FastAPI(title="Sentinel - Server Anomaly Detection API")

FEATURE_COLUMNS = [
    "feature_15",
    "feature_16",
    "feature_19",
    "feature_20",
    "feature_9",
]

TOTAL_REQUESTS = Counter(
    "total_requests",
    "Total number of POST /api/v1/analyze requests.",
)
ANOMALY_DETECTED_TOTAL = Counter(
    "anomaly_detected_total",
    "Total number of anomaly predictions.",
)


@app.on_event("startup")
def load_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "isolation_forest.pkl"
    scaler_path = project_root / "models" / "scaler.pkl"

    app.state.model = None
    app.state.scaler = None
    app.state.load_error = None

    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        with model_path.open("rb") as model_file:
            app.state.model = pickle.load(model_file)

        with scaler_path.open("rb") as scaler_file:
            app.state.scaler = pickle.load(scaler_file)
    except Exception as exc:  # noqa: BLE001
        app.state.load_error = str(exc)


@app.on_event("startup")
async def init_database() -> None:
    app.state.db_logging_error = None

    if not DB_LOGGING_AVAILABLE:
        app.state.db_logging_error = DB_LOGGING_ERROR
        return

    try:
        await init_db()
    except Exception as exc:  # noqa: BLE001
        app.state.db_logging_error = str(exc)


async def save_prediction_log(payload: dict[str, float], prediction_label: str) -> None:
    if not DB_LOGGING_AVAILABLE:
        return

    async with AsyncSessionLocal() as session:
        prediction_log = PredictionLog(
            feature_15=payload["feature_15"],
            feature_16=payload["feature_16"],
            feature_19=payload["feature_19"],
            feature_20=payload["feature_20"],
            feature_9=payload["feature_9"],
            prediction=prediction_label,
        )
        session.add(prediction_log)
        try:
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/v1/analyze")
def analyze_server_metrics(
    payload: ServerMetrics,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    TOTAL_REQUESTS.inc()

    if app.state.load_error is not None:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifacts unavailable: {app.state.load_error}",
        )

    if app.state.model is None or app.state.scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts are not loaded.",
        )

    try:
        values = [getattr(payload, feature) for feature in FEATURE_COLUMNS]
        input_array = np.array([values], dtype=float)

        scaled_input = app.state.scaler.transform(input_array)
        prediction = app.state.model.predict(scaled_input)[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        ) from exc

    prediction_label = "Normal" if prediction == 1 else "Anomaly"
    if prediction_label == "Anomaly":
        ANOMALY_DETECTED_TOTAL.inc()

    if DB_LOGGING_AVAILABLE and app.state.db_logging_error is None:
        background_tasks.add_task(
            save_prediction_log,
            payload.model_dump(),
            prediction_label,
        )
    else:
        app.state.db_logging_error = DB_LOGGING_ERROR

    return {
        "status": "success",
        "prediction": prediction_label,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
