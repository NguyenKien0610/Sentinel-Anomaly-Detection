from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.logging import get_logger
from src.db.models import PredictionLog


logger = get_logger("sentinel.prediction_log")


class PredictionLogService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None) -> None:
        self._session_factory = session_factory

    @property
    def enabled(self) -> bool:
        return self._session_factory is not None

    async def save_prediction_log(self, payload: dict[str, float], prediction_label: str) -> None:
        if self._session_factory is None:
            return

        async with self._session_factory() as session:
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
                logger.info(
                    "prediction_log_saved",
                    extra={
                        "extra_fields": {
                            "prediction": prediction_label,
                        }
                    },
                )
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                logger.exception(
                    "prediction_log_failed",
                    extra={
                        "extra_fields": {
                            "prediction": prediction_label,
                            "error": str(exc),
                        }
                    },
                )
