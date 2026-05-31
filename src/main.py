from fastapi import FastAPI

from src.api.routes import router
from src.core.config import get_settings
from src.core.logging import configure_logging
from src.db.session import database_manager
from src.middleware.request_context import RequestContextMiddleware
from src.queue import redis_manager
from src.services.drift_monitoring import DriftMonitoringService
from src.services.health import HealthService
from src.services.inference import InferenceService
from src.services.prediction_logging import PredictionLogService


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    inference_service = InferenceService(settings)
    drift_monitoring_service = DriftMonitoringService(settings)
    prediction_log_service = PredictionLogService(
        redis_client=redis_manager.client,
        stream_name=settings.prediction_log_stream,
    )
    health_service = HealthService(
        inference_service=inference_service,
        drift_monitoring_service=drift_monitoring_service,
        database_manager=database_manager,
        redis_manager=redis_manager,
    )

    app = FastAPI(title=settings.app_name)
    app.add_middleware(RequestContextMiddleware)
    app.state.settings = settings
    app.state.inference_service = inference_service
    app.state.drift_monitoring_service = drift_monitoring_service
    app.state.prediction_log_service = prediction_log_service
    app.state.database_manager = database_manager
    app.state.redis_manager = redis_manager
    app.state.health_service = health_service

    @app.on_event("startup")
    def load_application_artifacts() -> None:
        inference_service.load_artifacts()
        drift_monitoring_service.load_baseline()

    @app.on_event("shutdown")
    async def dispose_resources() -> None:
        await database_manager.dispose()
        await redis_manager.close()

    app.include_router(router)
    return app


app = create_app()
