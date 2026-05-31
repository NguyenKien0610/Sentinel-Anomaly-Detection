from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.logging import get_logger
from src.db.models import PredictionLog


logger = get_logger("sentinel.prediction_log")


class PredictionLogWriter:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None) -> None:
        self._session_factory = session_factory

    @property
    def enabled(self) -> bool:
        return self._session_factory is not None

    async def save_prediction_logs(self, events: list[dict[str, str]]) -> int:
        if self._session_factory is None:
            return 0

        prediction_logs = [
            PredictionLog(
                feature_15=float(event["feature_15"]),
                feature_16=float(event["feature_16"]),
                feature_19=float(event["feature_19"]),
                feature_20=float(event["feature_20"]),
                feature_9=float(event["feature_9"]),
                prediction=event["prediction"],
            )
            for event in events
        ]

        if not prediction_logs:
            return 0

        async with self._session_factory() as session:
            session.add_all(prediction_logs)
            try:
                await session.commit()
                logger.info(
                    "prediction_logs_saved",
                    extra={
                        "extra_fields": {
                            "count": len(prediction_logs),
                        }
                    },
                )
                return len(prediction_logs)
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                logger.exception(
                    "prediction_logs_failed",
                    extra={
                        "extra_fields": {
                            "count": len(prediction_logs),
                            "error": str(exc),
                        }
                    },
                )
                raise


class PredictionLogService:
    def __init__(
        self,
        redis_client: Redis | None,
        stream_name: str,
    ) -> None:
        self._redis_client = redis_client
        self._stream_name = stream_name

    @property
    def enabled(self) -> bool:
        return self._redis_client is not None

    async def enqueue_prediction_log(
        self,
        payload: dict[str, float],
        prediction_label: str,
        request_id: str,
    ) -> str | None:
        if self._redis_client is None:
            return None

        event = {
            "feature_15": str(payload["feature_15"]),
            "feature_16": str(payload["feature_16"]),
            "feature_19": str(payload["feature_19"]),
            "feature_20": str(payload["feature_20"]),
            "feature_9": str(payload["feature_9"]),
            "prediction": prediction_label,
            "request_id": request_id,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        try:
            event_id = await self._redis_client.xadd(self._stream_name, event)
            logger.info(
                "prediction_log_enqueued",
                extra={
                    "extra_fields": {
                        "stream": self._stream_name,
                        "event_id": event_id,
                        "prediction": prediction_label,
                    }
                },
            )
            return event_id
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "prediction_log_enqueue_failed",
                extra={
                    "extra_fields": {
                        "stream": self._stream_name,
                        "prediction": prediction_label,
                        "error": str(exc),
                    }
                },
            )
            return None
