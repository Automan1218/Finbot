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


@pytest_asyncio.fixture
async def finance_setup(db_session):
    import uuid as _uuid
    from app.core.security import hash_password, create_access_token
    from app.models.team import Team, TeamMember
    from app.models.user import User

    user = User(
        email=f"fin_{_uuid.uuid4().hex[:6]}@test.com",
        password_hash=hash_password("pw"),
    )
    db_session.add(user)
    await db_session.flush()

    team = Team(name="Test Fin Team", owner_id=user.id)
    db_session.add(team)
    await db_session.flush()

    member = TeamMember(team_id=team.id, user_id=user.id, role="owner")
    db_session.add(member)
    await db_session.commit()
    await db_session.refresh(user)
    await db_session.refresh(team)

    token = create_access_token(str(user.id))
    return user, team, token
