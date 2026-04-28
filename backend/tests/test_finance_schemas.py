from app.finance.schemas import (
    AccountCreate, AccountResponse, AccountUpdate,
    CategoryCreate, CategoryResponse,
    TransactionCreate, TransactionResponse, TransactionUpdate,
    BudgetCreate, BudgetResponse, BudgetUsage,
)
import uuid
from datetime import date


def test_account_create_defaults():
    a = AccountCreate(name="工行", type="bank", team_id=uuid.uuid4())
    assert a.currency == "CNY"
    assert a.balance_fen == 0


def test_transaction_create_requires_direction():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TransactionCreate(
            team_id=uuid.uuid4(),
            account_id=uuid.uuid4(),
            amount_fen=100,
            direction="invalid",
            transaction_date=date.today(),
        )


def test_budget_usage_ratio():
    u = BudgetUsage(
        budget_id=uuid.uuid4(),
        amount_fen=10000,
        spent_fen=8000,
        usage_ratio=0.8,
        period="monthly",
        period_start=date.today().replace(day=1),
        period_end=date.today(),
    )
    assert u.usage_ratio == 0.8
