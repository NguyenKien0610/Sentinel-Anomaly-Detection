from redis.asyncio import Redis

from src.core.config import get_settings


class RedisManager:
    def __init__(self, redis_url: str | None) -> None:
        self.redis_url = redis_url
        self.client: Redis | None = None

        if redis_url:
            self.client = Redis.from_url(redis_url, decode_responses=True)

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    async def healthcheck(self) -> tuple[bool, str | None]:
        if self.client is None:
            return False, "REDIS_URL environment variable is not configured."

        try:
            await self.client.ping()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

        return True, None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()


redis_manager = RedisManager(get_settings().redis_url)
