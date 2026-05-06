from datetime import date

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import create_account, create_category, create_transaction


def test_normalize_query_transactions_defaults_to_current_month():
    intent: AgentIntent = normalize_intent(
        "query_transactions",
        {},
        original_message="Show this month's expenses",
    )
    assert intent["name"] == "query_transactions"
    args = intent["arguments"]
    today = date.today()
    assert args["date_from"] == today.replace(day=1).isoformat()
    assert args["date_to"] == today.isoformat()
    assert args.get("direction") in {None, "expense", "income"}


def test_normalize_query_transactions_passes_filters():
    intent = normalize_intent(
        "query_transactions",
        {
            "date_from": "2026-04-01",
            "date_to": "2026-04-30",
            "direction": "expense",
            "category": "Food",
        },
        original_message="April food spending",
    )
    args = intent["arguments"]
    assert args["date_from"] == "2026-04-01"
    assert args["direction"] == "expense"
    assert args["category"] == "Food"


@pytest.mark.asyncio
async def test_execute_query_transactions_returns_filtered_rows(db_session, finance_setup):
    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "Food", None, None, db_session)
    travel = await create_category(team.id, "Travel", None, None, db_session)
    today = date.today()
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=food.id,
        amount_fen=3500,
        direction="expense",
        description="lunch",
        transaction_date=today,
        created_by=user.id,
        db=db_session,
    )
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=travel.id,
        amount_fen=4500,
        direction="expense",
        description="taxi",
        transaction_date=today,
        created_by=user.id,
        db=db_session,
    )

    intent: AgentIntent = {
        "name": "query_transactions",
        "arguments": {
            "date_from": today.isoformat(),
            "date_to": today.isoformat(),
            "category": "Food",
            "direction": "expense",
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "queried"
    assert result["count"] == 1
    assert result["total_fen"] == 3500
    assert len(result["items"]) == 1
    assert result["items"][0]["description"] == "lunch"
