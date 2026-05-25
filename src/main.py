import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

try:
    from src.schemas import ServerMetrics
except ModuleNotFoundError:
    from schemas import ServerMetrics


app = FastAPI(title="Sentinel - Server Anomaly Detection API")

FEATURE_COLUMNS = [
    "feature_15",
    "feature_16",
    "feature_19",
    "feature_20",
    "feature_9",
]


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


@app.post("/api/v1/analyze")
def analyze_server_metrics(payload: ServerMetrics) -> dict[str, Any]:
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

    return {
        "status": "success",
        "prediction": prediction_label,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
