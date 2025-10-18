from asyncio import current_task

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
    create_async_engine,
    async_scoped_session,
)

from .settings import settings


class DatabaseUtils:
    def __init__(self, url: str, echo: bool = True):
        self.engine = create_async_engine(url=url, echo=echo)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
        )

    async def get_scoped_session(self) -> AsyncSession:
        scoped_session = async_scoped_session(
            session_factory=self.session_factory, scopefunc=current_task
        )
        return scoped_session

    async def session_dependency(self) -> AsyncSession:
        session = await self.get_scoped_session()
        try:
            yield session
        finally:
            await session.close()


db_utils = DatabaseUtils(settings.database.async_url, echo=settings.database.echo)
