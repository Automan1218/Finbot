import uuid
from datetime import date

import pytest

from app.finance.service import (
    create_account,
    list_accounts,
    update_account,
    create_category,
    list_categories,
    create_transaction,
    list_transactions,
    soft_delete_transaction,
    create_budget,
    list_budgets,
    get_budget_usage,
)
from app.models.team import Team
from app.models.user import User


@pytest.fixture
async def setup(db_session):
    from app.core.security import hash_password
    user = User(email=f"svc_{uuid.uuid4().hex[:6]}@fin.com", password_hash=hash_password("pw"))
    db_session.add(user)
    await db_session.flush()
    team = Team(name="Fin", owner_id=user.id)
    db_session.add(team)
    await db_session.commit()
    await db_session.refresh(user)
    await db_session.refresh(team)
    return user, team


@pytest.mark.asyncio
async def test_create_and_list_accounts(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "工行卡", "bank", "CNY", 0, db_session)
    assert acc.id is not None
    assert acc.name == "工行卡"
    accounts = await list_accounts(team.id, db_session)
    assert any(a.id == acc.id for a in accounts)


@pytest.mark.asyncio
async def test_update_account(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "招行", "bank", "CNY", 0, db_session)
    updated = await update_account(acc.id, team.id, {"name": "招行储蓄卡", "balance_fen": 5000}, db_session)
    assert updated.name == "招行储蓄卡"
    assert updated.balance_fen == 5000


@pytest.mark.asyncio
async def test_create_and_list_categories(db_session, setup):
    user, team = setup
    cat = await create_category(team.id, "餐饮", "🍜", None, db_session)
    assert cat.name == "餐饮"
    cats = await list_categories(team.id, db_session)
    assert any(c.id == cat.id for c in cats)


@pytest.mark.asyncio
async def test_create_transaction(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    tx = await create_transaction(
        team_id=team.id,
        account_id=acc.id,
        category_id=None,
        amount_fen=3500,
        direction="expense",
        description="咖啡",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )
    assert tx.id is not None
    assert tx.amount_fen == 3500
    assert tx.deleted_at is None


@pytest.mark.asyncio
async def test_list_transactions_excludes_deleted(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "Card", "bank", "CNY", 0, db_session)
    tx = await create_transaction(
        team_id=team.id, account_id=acc.id, category_id=None,
        amount_fen=1000, direction="expense", description="test",
        transaction_date=date.today(), created_by=user.id, db=db_session,
    )
    await soft_delete_transaction(tx.id, team.id, db_session)
    txs = await list_transactions(team.id, None, None, db_session)
    assert all(t.id != tx.id for t in txs)


@pytest.mark.asyncio
async def test_budget_usage(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "Budget Acc", "bank", "CNY", 0, db_session)
    cat = await create_category(team.id, "餐饮", None, None, db_session)
    budget = await create_budget(team.id, cat.id, 10000, "monthly", 0.8, db_session)

    await create_transaction(
        team_id=team.id, account_id=acc.id, category_id=cat.id,
        amount_fen=3000, direction="expense", description="lunch",
        transaction_date=date.today(), created_by=user.id, db=db_session,
    )

    usage = await get_budget_usage(budget.id, team.id, db_session)
    assert usage["spent_fen"] == 3000
    assert usage["amount_fen"] == 10000
    assert abs(usage["usage_ratio"] - 0.3) < 0.001
