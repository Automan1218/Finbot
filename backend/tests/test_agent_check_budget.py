from datetime import date

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import (
    create_account,
    create_budget,
    create_category,
    create_transaction,
)


def test_normalize_check_budget_returns_intent():
    intent: AgentIntent = normalize_intent(
        "check_budget",
        {"category": "Food"},
        original_message="How much food budget is left?",
    )
    assert intent["name"] == "check_budget"
    assert intent["arguments"]["category"] == "Food"


def test_normalize_check_budget_without_category():
    intent = normalize_intent("check_budget", {}, original_message="Budget status")
    assert intent["name"] == "check_budget"
    assert intent["arguments"]["category"] is None


@pytest.mark.asyncio
async def test_execute_check_budget_returns_usage(db_session, finance_setup):
    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "Food", None, None, db_session)
    await create_budget(team.id, food.id, 100000, "monthly", 0.8, db_session)
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=food.id,
        amount_fen=20000,
        direction="expense",
        description="lunch",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )

    intent: AgentIntent = {
        "name": "check_budget",
        "arguments": {"category": "Food"},
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "checked"
    assert len(result["budgets"]) == 1
    item = result["budgets"][0]
    assert item["amount_fen"] == 100000
    assert item["spent_fen"] == 20000
    assert 0 < item["usage_ratio"] < 1


@pytest.mark.asyncio
async def test_execute_check_budget_no_match_returns_empty(db_session, finance_setup):
    user, team, _ = finance_setup
    intent: AgentIntent = {
        "name": "check_budget",
        "arguments": {"category": "Food"},
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "checked"
    assert result["budgets"] == []
