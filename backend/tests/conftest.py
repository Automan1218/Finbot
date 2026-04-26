import os

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://finbot:finbot@localhost:5433/finbot_test",
)


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    import app.models.account
    import app.models.alert
    import app.models.budget
    import app.models.category
    import app.models.experiment
    import app.models.feedback
    import app.models.knowledge
    import app.models.prompt_version
    import app.models.report
    import app.models.team
    import app.models.transaction
    import app.models.user
    from app.models.base import Base

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncSession:
    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
        await session.rollback()
