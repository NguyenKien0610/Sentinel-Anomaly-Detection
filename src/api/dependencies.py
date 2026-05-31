from fastapi import Request

from src.core.config import Settings
from src.services.drift_monitoring import DriftMonitoringService
from src.services.health import HealthService
from src.services.inference import InferenceService
from src.services.prediction_logging import PredictionLogService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def get_drift_monitoring_service(request: Request) -> DriftMonitoringService:
    return request.app.state.drift_monitoring_service


def get_prediction_log_service(request: Request) -> PredictionLogService:
    return request.app.state.prediction_log_service


def get_health_service(request: Request) -> HealthService:
    return request.app.state.health_service
