from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker, create_async_engine

from src.core.config import get_settings


class DatabaseManager:
    def __init__(self, database_url: str | None) -> None:
        self.database_url = database_url
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

        if database_url:
            self.engine = create_async_engine(database_url, future=True, pool_pre_ping=True)
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

    @property
    def is_configured(self) -> bool:
        return self.engine is not None and self.session_factory is not None

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        if self.session_factory is None:
            raise RuntimeError("DATABASE_URL environment variable is required.")

        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    async def healthcheck(self) -> tuple[bool, str | None]:
        if self.engine is None:
            return False, "DATABASE_URL environment variable is not configured."

        try:
            async with self.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

        return True, None


database_manager = DatabaseManager(get_settings().database_url)
engine = database_manager.engine
AsyncSessionLocal = database_manager.session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in database_manager.get_session():
        yield session
