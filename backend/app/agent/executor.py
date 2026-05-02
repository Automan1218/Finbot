import uuid
from datetime import date
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools import AgentIntent
from app.finance import service as finance_service
from app.models.account import Account
from app.models.budget import Budget
from app.models.category import Category
from app.models.transaction import Transaction
from app.rag.retrieve import smart_retrieve


async def execute_intent(
    intent: AgentIntent,
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    if intent["name"] == "record_transaction":
        return await _record_transaction(intent["arguments"], team_id, user_id, db)
    if intent["name"] == "generate_report":
        return await _generate_report(intent["arguments"], team_id, user_id, db)
    if intent["name"] == "rag_retrieve":
        return await _rag_retrieve(intent["arguments"], team_id, db)
    return {
        "status": "needs_clarification",
        "message": intent["arguments"]["question"],
        "missing_fields": intent["arguments"].get("missing_fields", []),
    }


async def _record_transaction(
    args: dict[str, Any],
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    account = await _resolve_account(team_id, args["account_name"], db)
    if account is None:
        return {
            "status": "needs_clarification",
            "message": "No active account exists for this team yet.",
            "missing_fields": ["account_id"],
        }

    category = await _resolve_category(team_id, args["category"], db)
    transaction = await finance_service.create_transaction(
        team_id=team_id,
        account_id=account.id,
        category_id=category.id if category else None,
        amount_fen=int(args["amount_fen"]),
        direction=args["direction"],
        description=args["description"],
        transaction_date=date.fromisoformat(args["transaction_date"]),
        created_by=user_id,
        db=db,
        created_by_ai=True,
    )
    alert_ids = await _check_budget_alerts(transaction, team_id, db)
    return {
        "status": "recorded",
        "message": "Transaction recorded.",
        "transaction_id": str(transaction.id),
        "account_id": str(account.id),
        "category_id": str(category.id) if category else None,
        "alert_ids": [str(alert_id) for alert_id in alert_ids],
    }


async def _generate_report(
    args: dict[str, Any],
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    period_start = date.fromisoformat(args["period_start"])
    period_end = date.fromisoformat(args["period_end"])
    group_by = args["group_by"]
    rows = await _aggregate_transactions(team_id, period_start, period_end, group_by, db)
    total_income_fen = sum(row["amount_fen"] for row in rows if row["direction"] == "income")
    total_expense_fen = sum(row["amount_fen"] for row in rows if row["direction"] == "expense")
    report = await finance_service.create_report(
        team_id=team_id,
        title=f"Finance report {period_start.isoformat()} to {period_end.isoformat()}",
        period_start=period_start,
        period_end=period_end,
        content=(
            f"Income: {total_income_fen} fen. "
            f"Expense: {total_expense_fen} fen. "
            f"Grouped by {group_by}."
        ),
        raw_data={
            "group_by": group_by,
            "rows": rows,
            "total_income_fen": total_income_fen,
            "total_expense_fen": total_expense_fen,
        },
        created_by=user_id,
        db=db,
    )
    return {
        "status": "reported",
        "message": "Report generated.",
        "report_id": str(report.id),
        "total_income_fen": total_income_fen,
        "total_expense_fen": total_expense_fen,
    }


async def _rag_retrieve(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    if not query:
        return {
            "status": "needs_clarification",
            "message": "Please describe what you want to look up.",
            "missing_fields": ["query"],
        }
    chunks = await smart_retrieve(query, team_id, db)
    return {
        "status": "retrieved",
        "message": f"Retrieved {len(chunks)} relevant chunks.",
        "chunks": [
            {"id": str(chunk.get("id")), "chunk_text": chunk.get("chunk_text")}
            for chunk in chunks
        ],
    }


async def _resolve_account(
    team_id: uuid.UUID, account_name: str, db: AsyncSession
) -> Account | None:
    result = await db.execute(
        select(Account).where(
            Account.team_id == team_id,
            Account.name == account_name,
            Account.is_active == True,
        )
    )
    account = result.scalar_one_or_none()
    if account:
        return account
    result = await db.execute(
        select(Account)
        .where(Account.team_id == team_id, Account.is_active == True)
        .order_by(Account.created_at.asc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _resolve_category(
    team_id: uuid.UUID, category_name: str, db: AsyncSession
) -> Category | None:
    result = await db.execute(
        select(Category)
        .where(
            Category.name == category_name,
            (Category.team_id == team_id) | Category.team_id.is_(None),
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _check_budget_alerts(
    transaction: Transaction, team_id: uuid.UUID, db: AsyncSession
) -> list[uuid.UUID]:
    if transaction.direction != "expense" or transaction.category_id is None:
        return []
    result = await db.execute(
        select(Budget).where(
            Budget.team_id == team_id,
            Budget.category_id == transaction.category_id,
            Budget.is_active == True,
        )
    )
    alert_ids: list[uuid.UUID] = []
    for budget in result.scalars().all():
        usage = await finance_service.get_budget_usage(budget.id, team_id, db)
        if usage["usage_ratio"] >= budget.alert_threshold:
            alert = await finance_service.create_alert(
                team_id=team_id,
                budget_id=budget.id,
                triggered_by=transaction.id,
                usage_ratio=usage["usage_ratio"],
                message=(
                    f"Budget usage reached {usage['usage_ratio']:.0%} "
                    f"for budget {budget.id}."
                ),
                db=db,
            )
            alert_ids.append(alert.id)
    return alert_ids


async def _aggregate_transactions(
    team_id: uuid.UUID,
    period_start: date,
    period_end: date,
    group_by: str,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    group_column = {
        "category": Transaction.category_id,
        "account": Transaction.account_id,
        "day": Transaction.transaction_date,
    }[group_by]
    result = await db.execute(
        select(
            group_column.label("group_key"),
            Transaction.direction,
            func.coalesce(func.sum(Transaction.amount_fen), 0).label("amount_fen"),
        )
        .where(
            and_(
                Transaction.team_id == team_id,
                Transaction.transaction_date >= period_start,
                Transaction.transaction_date <= period_end,
                Transaction.deleted_at.is_(None),
            )
        )
        .group_by(group_column, Transaction.direction)
        .order_by(group_column)
    )
    rows = []
    for group_key, direction, amount_fen in result.all():
        rows.append(
            {
                "group_key": group_key.isoformat() if isinstance(group_key, date) else str(group_key),
                "direction": direction,
                "amount_fen": int(amount_fen),
            }
        )
    return rows
