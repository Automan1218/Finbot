import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import create_account


def test_normalize_record_batch_keeps_three_items():
    intent: AgentIntent = normalize_intent(
        "record_batch",
        {
            "transactions": [
                {
                    "amount_yuan": 12,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "breakfast",
                },
                {
                    "amount_yuan": 35,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "lunch",
                },
                {
                    "amount_yuan": 45,
                    "direction": "expense",
                    "category": "Transport",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "taxi",
                },
            ]
        },
        original_message="Today breakfast 12 lunch 35 taxi 45",
    )
    assert intent["name"] == "record_batch"
    items = intent["arguments"]["transactions"]
    assert len(items) == 3
    assert items[0]["amount_fen"] == 1200
    assert items[2]["category"] == "Transport"


def test_normalize_record_batch_falls_back_to_clarify_when_empty():
    intent = normalize_intent("record_batch", {"transactions": []}, original_message="")
    assert intent["name"] == "clarify"


def test_normalize_record_batch_drops_invalid_items():
    intent = normalize_intent(
        "record_batch",
        {
            "transactions": [
                {
                    "amount_yuan": 12,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "breakfast",
                },
                {"amount_yuan": None, "direction": "expense"},
            ]
        },
        original_message="Today breakfast 12",
    )
    assert intent["name"] == "record_batch"
    assert len(intent["arguments"]["transactions"]) == 1


@pytest.mark.asyncio
async def test_execute_record_batch_creates_three_transactions(db_session, finance_setup):
    user, team, _ = finance_setup
    await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)

    intent: AgentIntent = {
        "name": "record_batch",
        "arguments": {
            "transactions": [
                {
                    "amount_yuan": 12.0,
                    "amount_fen": 1200,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "breakfast",
                },
                {
                    "amount_yuan": 35.0,
                    "amount_fen": 3500,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "lunch",
                },
                {
                    "amount_yuan": 45.0,
                    "amount_fen": 4500,
                    "direction": "expense",
                    "category": "Transport",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "taxi",
                },
            ]
        },
    }

    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "recorded_batch"
    assert result["count"] == 3
    assert len(result["transaction_ids"]) == 3


@pytest.mark.asyncio
async def test_execute_record_batch_no_account_returns_clarification(db_session, finance_setup):
    user, team, _ = finance_setup
    intent: AgentIntent = {
        "name": "record_batch",
        "arguments": {
            "transactions": [
                {
                    "amount_yuan": 12.0,
                    "amount_fen": 1200,
                    "direction": "expense",
                    "category": "Food",
                    "account_name": "Cash",
                    "transaction_date": "2026-05-05",
                    "description": "breakfast",
                },
            ]
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "needs_clarification"
