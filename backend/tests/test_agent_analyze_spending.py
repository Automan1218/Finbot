from datetime import date, timedelta

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import create_account, create_category, create_transaction


def test_normalize_analyze_spending_defaults():
    intent: AgentIntent = normalize_intent(
        "analyze_spending",
        {},
        original_message="How much did we spend recently?",
    )
    args = intent["arguments"]
    assert intent["name"] == "analyze_spending"
    assert args["period"] == "last_30_days"
    assert args["group_by"] == "category"
    assert args["compare_with"] in {"prev_period", None}


def test_normalize_analyze_spending_custom_dates():
    intent = normalize_intent(
        "analyze_spending",
        {
            "period": "custom",
            "period_start": "2026-04-01",
            "period_end": "2026-04-30",
            "group_by": "day",
            "compare_with": "prev_period",
        },
        original_message="April daily spending",
    )
    args = intent["arguments"]
    assert args["period"] == "custom"
    assert args["period_start"] == "2026-04-01"
    assert args["compare_with"] == "prev_period"


@pytest.mark.asyncio
async def test_execute_analyze_spending_compares_two_periods(db_session, finance_setup):
    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "Food", None, None, db_session)

    today = date.today()
    cur_start = today - timedelta(days=10)
    prev_start = today - timedelta(days=20)

    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=food.id,
        amount_fen=10000,
        direction="expense",
        description="cur",
        transaction_date=cur_start,
        created_by=user.id,
        db=db_session,
    )
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=food.id,
        amount_fen=4000,
        direction="expense",
        description="prev",
        transaction_date=prev_start,
        created_by=user.id,
        db=db_session,
    )

    intent: AgentIntent = {
        "name": "analyze_spending",
        "arguments": {
            "period": "custom",
            "period_start": cur_start.isoformat(),
            "period_end": today.isoformat(),
            "group_by": "category",
            "compare_with": "prev_period",
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "analyzed"
    assert result["current_total_fen"] == 10000
    assert result["compare_total_fen"] == 4000
    assert result["delta_fen"] == 6000
    assert any(r["amount_fen"] == 10000 for r in result["current_rows"])


@pytest.mark.asyncio
async def test_execute_analyze_spending_without_compare(db_session, finance_setup):
    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "Food", None, None, db_session)
    today = date.today()
    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=food.id,
        amount_fen=5000,
        direction="expense",
        description="x",
        transaction_date=today,
        created_by=user.id,
        db=db_session,
    )

    intent: AgentIntent = {
        "name": "analyze_spending",
        "arguments": {
            "period": "custom",
            "period_start": today.isoformat(),
            "period_end": today.isoformat(),
            "group_by": "category",
            "compare_with": None,
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "analyzed"
    assert result["compare_total_fen"] is None
    assert result["delta_fen"] is None
