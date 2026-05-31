from dataclasses import dataclass

from src.db.session import DatabaseManager
from src.queue import RedisManager
from src.services.drift_monitoring import DriftMonitoringService
from src.services.inference import InferenceService


@dataclass(frozen=True)
class HealthCheckResult:
    status: str
    checks: dict[str, dict[str, object]]


class HealthService:
    def __init__(
        self,
        inference_service: InferenceService,
        drift_monitoring_service: DriftMonitoringService,
        database_manager: DatabaseManager,
        redis_manager: RedisManager,
    ) -> None:
        self._inference_service = inference_service
        self._drift_monitoring_service = drift_monitoring_service
        self._database_manager = database_manager
        self._redis_manager = redis_manager

    def liveness(self) -> dict[str, str]:
        return {"status": "ok"}

    async def readiness(self) -> HealthCheckResult:
        db_ok, db_error = await self._database_manager.healthcheck()
        redis_ok, redis_error = await self._redis_manager.healthcheck()

        checks = {
            "model": {
                "ready": self._inference_service.load_error is None
                and self._inference_service.model is not None
                and self._inference_service.scaler is not None,
                "error": self._inference_service.load_error,
            },
            "drift_monitoring": {
                "ready": self._drift_monitoring_service.baseline_ready,
                "error": self._drift_monitoring_service.load_error,
            },
            "database": {
                "ready": db_ok,
                "error": db_error,
            },
            "redis": {
                "ready": redis_ok,
                "error": redis_error,
            },
        }

        status = "ok" if all(check["ready"] for check in checks.values()) else "degraded"
        return HealthCheckResult(status=status, checks=checks)
