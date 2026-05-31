import asyncio

from redis.exceptions import ResponseError

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.db.session import AsyncSessionLocal, database_manager
from src.queue import redis_manager
from src.services.prediction_logging import PredictionLogWriter


logger = get_logger("sentinel.prediction_log_worker")


class PredictionLogWorker:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = redis_manager.client
        self._writer = PredictionLogWriter(AsyncSessionLocal)

    async def run(self) -> None:
        if self._redis is None:
            raise RuntimeError("REDIS_URL environment variable is required for worker.")
        if not self._writer.enabled:
            raise RuntimeError("DATABASE_URL environment variable is required for worker.")

        await self._ensure_consumer_group()
        logger.info(
            "prediction_log_worker_started",
            extra={
                "extra_fields": {
                    "stream": self._settings.prediction_log_stream,
                    "group": self._settings.prediction_log_group,
                    "consumer": self._settings.prediction_log_consumer,
                    "batch_size": self._settings.prediction_log_batch_size,
                }
            },
        )

        while True:
            response = await self._redis.xreadgroup(
                groupname=self._settings.prediction_log_group,
                consumername=self._settings.prediction_log_consumer,
                streams={self._settings.prediction_log_stream: ">"},
                count=self._settings.prediction_log_batch_size,
                block=5000,
            )
            if not response:
                continue

            for _, messages in response:
                event_ids = [event_id for event_id, _ in messages]
                events = [event for _, event in messages]
                await self._writer.save_prediction_logs(events)
                await self._redis.xack(
                    self._settings.prediction_log_stream,
                    self._settings.prediction_log_group,
                    *event_ids,
                )
                logger.info(
                    "prediction_log_events_acked",
                    extra={
                        "extra_fields": {
                            "count": len(event_ids),
                        }
                    },
                )

    async def _ensure_consumer_group(self) -> None:
        try:
            await self._redis.xgroup_create(
                name=self._settings.prediction_log_stream,
                groupname=self._settings.prediction_log_group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise


async def main() -> None:
    configure_logging()
    try:
        await PredictionLogWorker().run()
    finally:
        await redis_manager.close()
        await database_manager.dispose()


if __name__ == "__main__":
    asyncio.run(main())
