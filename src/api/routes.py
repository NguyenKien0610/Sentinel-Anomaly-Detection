from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from src.api.dependencies import (
    get_drift_monitoring_service,
    get_health_service,
    get_inference_service,
    get_prediction_log_service,
    get_settings,
)
from src.core.logging import get_logger
from src.schemas import ServerMetrics
from src.services.metrics import (
    build_metrics_response,
    record_anomaly,
    record_drift_snapshot,
    record_prediction_label,
    record_request,
    record_request_latency,
)


router = APIRouter()
logger = get_logger("sentinel.api")


@router.get("/", include_in_schema=False)
def frontend(request: Request) -> FileResponse:
    settings = get_settings(request)
    return FileResponse(settings.static_index_path)


@router.get("/metrics")
def metrics():
    return build_metrics_response()


@router.get("/healthz")
def healthz(request: Request) -> dict[str, str]:
    health_service = get_health_service(request)
    return health_service.liveness()


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    health_service = get_health_service(request)
    readiness = await health_service.readiness()
    status_code = 200 if readiness.status == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": readiness.status,
            "checks": readiness.checks,
        },
    )


@router.get("/api/v1/model/info")
def model_info(request: Request) -> dict[str, object]:
    inference_service = get_inference_service(request)
    return inference_service.model_info()


@router.get("/api/v1/monitoring/drift")
def drift_status(request: Request) -> dict[str, object]:
    drift_monitoring_service = get_drift_monitoring_service(request)
    return drift_monitoring_service.current_status()


@router.post("/api/v1/analyze")
def analyze_server_metrics(
    payload: ServerMetrics,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict[str, Any]:
    started_at = perf_counter()
    record_request()

    inference_service = get_inference_service(request)
    drift_monitoring_service = get_drift_monitoring_service(request)
    prediction_log_service = get_prediction_log_service(request)
    feature_payload = payload.to_feature_payload()

    try:
        prediction_label = inference_service.predict(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        record_request_latency(perf_counter() - started_at)

    record_prediction_label(prediction_label)
    if prediction_label == "Anomaly":
        record_anomaly()

    drift_snapshot = drift_monitoring_service.evaluate(feature_payload)
    if drift_snapshot is not None:
        record_drift_snapshot(
            score=drift_snapshot.score,
            threshold=drift_snapshot.threshold,
            per_feature_abs_zscore=drift_snapshot.per_feature_abs_zscore,
            values=drift_snapshot.values,
            alert=drift_snapshot.alert,
        )

    logger.info(
        "inference_completed",
        extra={
            "extra_fields": {
                "prediction": prediction_label,
                "drift_score": None if drift_snapshot is None else drift_snapshot.score,
                "drift_alert": None if drift_snapshot is None else drift_snapshot.alert,
                "path": request.url.path,
            }
        },
    )

    if prediction_log_service.enabled:
        background_tasks.add_task(
            prediction_log_service.save_prediction_log,
            feature_payload,
            prediction_label,
        )

    return {
        "status": "success",
        "prediction": prediction_label,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
