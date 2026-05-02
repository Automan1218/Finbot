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
    count_transactions,
    get_transaction,
    list_transactions,
    soft_delete_transaction,
    create_budget,
    list_budgets,
    update_budget,
    delete_budget,
    get_budget_usage,
    create_alert,
    list_alerts,
    mark_alert_read,
    create_report,
    list_reports,
    get_report,
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
async def test_get_transaction_and_filtered_count(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "Filtered Card", "bank", "CNY", 0, db_session)
    other_acc = await create_account(team.id, "Other Card", "bank", "CNY", 0, db_session)
    cat = await create_category(team.id, "Filtered Cat", None, None, db_session)
    tx = await create_transaction(
        team_id=team.id,
        account_id=acc.id,
        category_id=cat.id,
        amount_fen=1200,
        direction="expense",
        description="filtered",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )
    await create_transaction(
        team_id=team.id,
        account_id=other_acc.id,
        category_id=None,
        amount_fen=900,
        direction="expense",
        description="other",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
    )

    fetched = await get_transaction(tx.id, team.id, db_session)
    filtered = await list_transactions(
        team.id, None, None, db_session, category_id=cat.id, account_id=acc.id
    )
    total = await count_transactions(
        team.id, None, None, db_session, category_id=cat.id, account_id=acc.id
    )
    assert fetched.id == tx.id
    assert [item.id for item in filtered] == [tx.id]
    assert total == 1


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


@pytest.mark.asyncio
async def test_update_and_delete_budget(db_session, setup):
    user, team = setup
    cat = await create_category(team.id, "Budget Update", None, None, db_session)
    budget = await create_budget(team.id, cat.id, 10000, "monthly", 0.8, db_session)

    updated = await update_budget(
        budget.id, team.id, {"amount_fen": 15000, "alert_threshold": 0.9}, db_session
    )
    assert updated.amount_fen == 15000
    assert updated.alert_threshold == 0.9

    await delete_budget(budget.id, team.id, db_session)
    budgets = await list_budgets(team.id, db_session)
    assert all(item.id != budget.id for item in budgets)


@pytest.mark.asyncio
async def test_alert_lifecycle(db_session, setup):
    user, team = setup
    acc = await create_account(team.id, "Alert Acc", "bank", "CNY", 0, db_session)
    cat = await create_category(team.id, "Alert Cat", None, None, db_session)
    budget = await create_budget(team.id, cat.id, 10000, "monthly", 0.8, db_session)
    tx = await create_transaction(
        team_id=team.id, account_id=acc.id, category_id=cat.id,
        amount_fen=9000, direction="expense", description="alert",
        transaction_date=date.today(), created_by=user.id, db=db_session,
    )

    alert = await create_alert(team.id, budget.id, tx.id, 0.9, "Budget warning", db_session)
    unread = await list_alerts(team.id, False, db_session)
    assert any(item.id == alert.id for item in unread)

    read = await mark_alert_read(alert.id, team.id, db_session)
    assert read.is_read is True


@pytest.mark.asyncio
async def test_report_lifecycle(db_session, setup):
    user, team = setup
    report = await create_report(
        team_id=team.id,
        title="Monthly",
        period_start=date.today(),
        period_end=date.today(),
        content="summary",
        raw_data={"total_expense_fen": 1000},
        created_by=user.id,
        db=db_session,
    )

    reports = await list_reports(team.id, db_session)
    assert any(item.id == report.id for item in reports)
    fetched = await get_report(report.id, team.id, db_session)
    assert fetched.raw_data["total_expense_fen"] == 1000
