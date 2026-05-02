from datetime import date
from unittest.mock import AsyncMock
import uuid

import pytest

from app.agent.executor import execute_intent
from app.agent.tools import detect_intent
from app.finance.service import (
    create_account,
    create_budget,
    create_category,
    create_transaction,
)
from app.models.team import Team
from app.models.user import User


@pytest.fixture
async def setup(db_session):
    from app.core.security import hash_password

    user = User(email=f"agent_exec_{uuid.uuid4().hex[:8]}@test.com", password_hash=hash_password("pw"))
    db_session.add(user)
    await db_session.flush()
    team = Team(name="Agent Exec", owner_id=user.id)
    db_session.add(team)
    await db_session.commit()
    await db_session.refresh(user)
    await db_session.refresh(team)
    return user, team


@pytest.mark.asyncio
async def test_execute_record_transaction_writes_ai_transaction(db_session, setup):
    user, team = setup
    account = await create_account(team.id, "支付宝", "cash", "CNY", 0, db_session)
    category = await create_category(team.id, "餐饮", None, None, db_session)
    intent = detect_intent("今天午饭花了 35 元，用支付宝")

    result = await execute_intent(intent, team.id, user.id, db_session)

    assert result["status"] == "recorded"
    assert result["account_id"] == str(account.id)
    assert result["category_id"] == str(category.id)
    assert result["transaction_id"]


@pytest.mark.asyncio
async def test_execute_record_transaction_creates_budget_alert(db_session, setup):
    user, team = setup
    await create_account(team.id, "支付宝", "cash", "CNY", 0, db_session)
    category = await create_category(team.id, "餐饮", None, None, db_session)
    await create_budget(team.id, category.id, 10000, "monthly", 0.8, db_session)
    await create_transaction(
        team_id=team.id,
        account_id=(await create_account(team.id, "Reserve", "cash", "CNY", 0, db_session)).id,
        category_id=category.id,
        amount_fen=5000,
        direction="expense",
        description="existing",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )
    intent = detect_intent("今天午饭花了 35 元，用支付宝")

    result = await execute_intent(intent, team.id, user.id, db_session)

    assert result["status"] == "recorded"
    assert len(result["alert_ids"]) == 1


@pytest.mark.asyncio
async def test_execute_generate_report_creates_report(db_session, setup):
    user, team = setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=None,
        amount_fen=1200,
        direction="expense",
        description="coffee",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )
    intent = {
        "name": "generate_report",
        "arguments": {
            "period_start": date.today().replace(day=1).isoformat(),
            "period_end": date.today().isoformat(),
            "group_by": "account",
        },
    }

    result = await execute_intent(intent, team.id, user.id, db_session)

    assert result["status"] == "reported"
    assert result["report_id"]
    assert result["total_expense_fen"] == 1200


@pytest.mark.asyncio
async def test_execute_rag_retrieve_returns_chunks(db_session, setup, monkeypatch):
    user, team = setup
    fake_retrieve = AsyncMock(
        return_value=[{"id": "doc-1", "chunk_text": "policy snippet"}]
    )
    monkeypatch.setattr("app.agent.executor.smart_retrieve", fake_retrieve)

    intent = {"name": "rag_retrieve", "arguments": {"query": "出差报销"}}
    result = await execute_intent(intent, team.id, user.id, db_session)

    assert result["status"] == "retrieved"
    assert result["chunks"][0]["chunk_text"] == "policy snippet"
    fake_retrieve.assert_awaited_once()
