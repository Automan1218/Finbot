import pytest
from sqlalchemy import text
from app.core.database import get_db_session
from app.core.redis import get_redis


@pytest.mark.asyncio
async def test_postgres_connectivity():
    async with get_db_session() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_pgvector_extension():
    async with get_db_session() as session:
        result = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        assert result.scalar() == "vector", "pgvector extension not installed"


@pytest.mark.asyncio
async def test_redis_connectivity():
    r = await get_redis()
    await r.set("test:ping", "pong", ex=10)
    val = await r.get("test:ping")
    assert val == b"pong"
    await r.delete("test:ping")


def test_model_imports():
    from app.models.user import User
    from app.models.team import Team, TeamMember
    assert User.__tablename__ == "users"
    assert Team.__tablename__ == "teams"
    assert TeamMember.__tablename__ == "team_members"


def test_financial_model_imports():
    from app.models.account import Account
    from app.models.category import Category
    from app.models.transaction import Transaction
    from app.models.budget import Budget
    from app.models.alert import Alert
    from app.models.report import Report
    assert Account.__tablename__ == "accounts"
    assert Category.__tablename__ == "categories"
    assert Transaction.__tablename__ == "transactions"
    assert Budget.__tablename__ == "budgets"
    assert Alert.__tablename__ == "alerts"
    assert Report.__tablename__ == "reports"


def test_ai_model_imports():
    from app.models.feedback import Feedback
    from app.models.prompt_version import PromptVersion
    from app.models.experiment import Experiment
    from app.models.knowledge import Document, DocumentChunk
    assert Feedback.__tablename__ == "feedback"
    assert PromptVersion.__tablename__ == "prompt_versions"
    assert Experiment.__tablename__ == "experiments"
    assert Document.__tablename__ == "documents"
    assert DocumentChunk.__tablename__ == "document_chunks"
