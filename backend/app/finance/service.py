import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.account import Account
from app.models.alert import Alert
from app.models.budget import Budget
from app.models.category import Category
from app.models.report import Report
from app.models.transaction import Transaction


async def create_account(
    team_id: uuid.UUID, name: str, type: str, currency: str, balance_fen: int, db: AsyncSession
) -> Account:
    acc = Account(team_id=team_id, name=name, type=type, currency=currency, balance_fen=balance_fen)
    db.add(acc)
    await db.commit()
    await db.refresh(acc)
    return acc


async def list_accounts(team_id: uuid.UUID, db: AsyncSession) -> list[Account]:
    result = await db.execute(
        select(Account).where(Account.team_id == team_id, Account.is_active == True)
    )
    return list(result.scalars().all())


async def update_account(
    account_id: uuid.UUID, team_id: uuid.UUID, fields: dict[str, Any], db: AsyncSession
) -> Account:
    result = await db.execute(
        select(Account).where(Account.id == account_id, Account.team_id == team_id)
    )
    acc = result.scalar_one_or_none()
    if not acc:
        raise ValueError("Account not found")
    for key, val in fields.items():
        if val is not None:
            setattr(acc, key, val)
    await db.commit()
    await db.refresh(acc)
    return acc


async def create_category(
    team_id: uuid.UUID, name: str, icon: str | None, parent_id: uuid.UUID | None, db: AsyncSession
) -> Category:
    cat = Category(team_id=team_id, name=name, icon=icon, parent_id=parent_id)
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return cat


async def list_categories(team_id: uuid.UUID, db: AsyncSession) -> list[Category]:
    result = await db.execute(
        select(Category).where(
            or_(Category.team_id == team_id, Category.team_id.is_(None))
        )
    )
    return list(result.scalars().all())


async def create_transaction(
    team_id: uuid.UUID,
    account_id: uuid.UUID,
    category_id: uuid.UUID | None,
    amount_fen: int,
    direction: str,
    description: str | None,
    transaction_date: date,
    created_by: uuid.UUID,
    db: AsyncSession,
    created_by_ai: bool = False,
) -> Transaction:
    tx = Transaction(
        team_id=team_id,
        account_id=account_id,
        category_id=category_id,
        amount_fen=amount_fen,
        direction=direction,
        description=description,
        transaction_date=transaction_date,
        created_by=created_by,
        created_by_ai=created_by_ai,
    )
    db.add(tx)
    await db.commit()
    await db.refresh(tx)
    return tx


async def list_transactions(
    team_id: uuid.UUID,
    date_from: date | None,
    date_to: date | None,
    db: AsyncSession,
    category_id: uuid.UUID | None = None,
    account_id: uuid.UUID | None = None,
    page: int | None = None,
    size: int | None = None,
) -> list[Transaction]:
    filters = _transaction_filters(team_id, date_from, date_to, category_id, account_id)
    stmt = select(Transaction).where(and_(*filters)).order_by(Transaction.transaction_date.desc())
    if page is not None and size is not None:
        stmt = stmt.offset((page - 1) * size).limit(size)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def count_transactions(
    team_id: uuid.UUID,
    date_from: date | None,
    date_to: date | None,
    db: AsyncSession,
    category_id: uuid.UUID | None = None,
    account_id: uuid.UUID | None = None,
) -> int:
    filters = _transaction_filters(team_id, date_from, date_to, category_id, account_id)
    result = await db.execute(
        select(func.count()).select_from(Transaction).where(and_(*filters))
    )
    return int(result.scalar() or 0)


async def get_transaction(
    tx_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession
) -> Transaction:
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == tx_id,
            Transaction.team_id == team_id,
            Transaction.deleted_at.is_(None),
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise ValueError("Transaction not found")
    return tx


async def update_transaction(
    tx_id: uuid.UUID, team_id: uuid.UUID, fields: dict[str, Any], db: AsyncSession
) -> Transaction:
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == tx_id,
            Transaction.team_id == team_id,
            Transaction.deleted_at.is_(None),
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise ValueError("Transaction not found")
    for key, val in fields.items():
        if val is not None:
            setattr(tx, key, val)
    await db.commit()
    await db.refresh(tx)
    return tx


def _transaction_filters(
    team_id: uuid.UUID,
    date_from: date | None,
    date_to: date | None,
    category_id: uuid.UUID | None,
    account_id: uuid.UUID | None,
) -> list[Any]:
    filters = [Transaction.team_id == team_id, Transaction.deleted_at.is_(None)]
    if date_from:
        filters.append(Transaction.transaction_date >= date_from)
    if date_to:
        filters.append(Transaction.transaction_date <= date_to)
    if category_id:
        filters.append(Transaction.category_id == category_id)
    if account_id:
        filters.append(Transaction.account_id == account_id)
    return filters


async def soft_delete_transaction(
    tx_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession
) -> None:
    result = await db.execute(
        select(Transaction).where(
            Transaction.id == tx_id,
            Transaction.team_id == team_id,
            Transaction.deleted_at.is_(None),
        )
    )
    tx = result.scalar_one_or_none()
    if not tx:
        raise ValueError("Transaction not found")
    tx.deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()


async def create_budget(
    team_id: uuid.UUID, category_id: uuid.UUID, amount_fen: int,
    period: str, alert_threshold: float, db: AsyncSession
) -> Budget:
    budget = Budget(
        team_id=team_id, category_id=category_id, amount_fen=amount_fen,
        period=period, alert_threshold=alert_threshold,
    )
    db.add(budget)
    await db.commit()
    await db.refresh(budget)
    return budget


async def list_budgets(team_id: uuid.UUID, db: AsyncSession) -> list[Budget]:
    result = await db.execute(
        select(Budget).where(Budget.team_id == team_id, Budget.is_active == True)
    )
    return list(result.scalars().all())


async def update_budget(
    budget_id: uuid.UUID, team_id: uuid.UUID, fields: dict[str, Any], db: AsyncSession
) -> Budget:
    result = await db.execute(
        select(Budget).where(Budget.id == budget_id, Budget.team_id == team_id)
    )
    budget = result.scalar_one_or_none()
    if not budget:
        raise ValueError("Budget not found")
    for key, val in fields.items():
        if val is not None:
            setattr(budget, key, val)
    await db.commit()
    await db.refresh(budget)
    return budget


async def delete_budget(budget_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession) -> None:
    result = await db.execute(
        select(Budget).where(
            Budget.id == budget_id,
            Budget.team_id == team_id,
            Budget.is_active == True,
        )
    )
    budget = result.scalar_one_or_none()
    if not budget:
        raise ValueError("Budget not found")
    budget.is_active = False
    await db.commit()


async def get_budget_usage(
    budget_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession
) -> dict:
    from datetime import date as date_type
    result = await db.execute(
        select(Budget).where(Budget.id == budget_id, Budget.team_id == team_id)
    )
    budget = result.scalar_one_or_none()
    if not budget:
        raise ValueError("Budget not found")

    today = date_type.today()
    if budget.period == "monthly":
        period_start = today.replace(day=1)
        if today.month == 12:
            period_end = date_type(today.year + 1, 1, 1)
        else:
            period_end = date_type(today.year, today.month + 1, 1)
    else:
        quarter = (today.month - 1) // 3
        period_start = date_type(today.year, quarter * 3 + 1, 1)
        if quarter == 3:
            period_end = date_type(today.year + 1, 1, 1)
        else:
            period_end = date_type(today.year, (quarter + 1) * 3 + 1, 1)

    spent_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount_fen), 0)).where(
            and_(
                Transaction.team_id == team_id,
                Transaction.category_id == budget.category_id,
                Transaction.direction == "expense",
                Transaction.transaction_date >= period_start,
                Transaction.transaction_date < period_end,
                Transaction.deleted_at.is_(None),
            )
        )
    )
    spent_fen = int(spent_result.scalar() or 0)

    return {
        "budget_id": budget_id,
        "amount_fen": budget.amount_fen,
        "spent_fen": spent_fen,
        "usage_ratio": spent_fen / budget.amount_fen if budget.amount_fen > 0 else 0.0,
        "period": budget.period,
        "period_start": period_start,
        "period_end": period_end,
    }


async def create_alert(
    team_id: uuid.UUID,
    budget_id: uuid.UUID,
    triggered_by: uuid.UUID,
    usage_ratio: float,
    message: str | None,
    db: AsyncSession,
) -> Alert:
    alert = Alert(
        team_id=team_id,
        budget_id=budget_id,
        triggered_by=triggered_by,
        usage_ratio=usage_ratio,
        message=message,
    )
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    return alert


async def list_alerts(
    team_id: uuid.UUID, is_read: bool | None, db: AsyncSession
) -> list[Alert]:
    filters = [Alert.team_id == team_id]
    if is_read is not None:
        filters.append(Alert.is_read == is_read)
    result = await db.execute(
        select(Alert).where(and_(*filters)).order_by(Alert.created_at.desc())
    )
    return list(result.scalars().all())


async def mark_alert_read(alert_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession) -> Alert:
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id, Alert.team_id == team_id)
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise ValueError("Alert not found")
    alert.is_read = True
    await db.commit()
    await db.refresh(alert)
    return alert


async def create_report(
    team_id: uuid.UUID,
    title: str | None,
    period_start: date,
    period_end: date,
    content: str | None,
    raw_data: dict[str, Any] | None,
    created_by: uuid.UUID,
    db: AsyncSession,
) -> Report:
    report = Report(
        team_id=team_id,
        title=title,
        period_start=period_start,
        period_end=period_end,
        content=content,
        raw_data=raw_data,
        created_by=created_by,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return report


async def list_reports(team_id: uuid.UUID, db: AsyncSession) -> list[Report]:
    result = await db.execute(
        select(Report).where(Report.team_id == team_id).order_by(Report.created_at.desc())
    )
    return list(result.scalars().all())


async def get_report(report_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession) -> Report:
    result = await db.execute(
        select(Report).where(Report.id == report_id, Report.team_id == team_id)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise ValueError("Report not found")
    return report
