import uuid
from datetime import date, timedelta
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
    if intent["name"] == "record_batch":
        return await _record_batch(intent["arguments"], team_id, user_id, db)
    if intent["name"] == "query_transactions":
        return await _query_transactions(intent["arguments"], team_id, db)
    if intent["name"] == "analyze_spending":
        return await _analyze_spending(intent["arguments"], team_id, db)
    if intent["name"] == "check_budget":
        return await _check_budget(intent["arguments"], team_id, db)
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


async def _record_batch(
    args: dict[str, Any],
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    items = args.get("transactions") or []
    if not items:
        return {
            "status": "needs_clarification",
            "message": "No transactions provided.",
            "missing_fields": ["transactions"],
        }

    transaction_ids: list[str] = []
    alert_ids: list[str] = []
    for item in items:
        account = await _resolve_account(team_id, item["account_name"], db)
        if account is None:
            return {
                "status": "needs_clarification",
                "message": "No active account exists for this team yet.",
                "missing_fields": ["account_id"],
            }
        category = await _resolve_category(team_id, item["category"], db)
        transaction = await finance_service.create_transaction(
            team_id=team_id,
            account_id=account.id,
            category_id=category.id if category else None,
            amount_fen=int(item["amount_fen"]),
            direction=item["direction"],
            description=item["description"],
            transaction_date=date.fromisoformat(item["transaction_date"]),
            created_by=user_id,
            db=db,
            created_by_ai=True,
        )
        transaction_ids.append(str(transaction.id))
        for alert_id in await _check_budget_alerts(transaction, team_id, db):
            alert_ids.append(str(alert_id))

    return {
        "status": "recorded_batch",
        "message": f"Recorded {len(transaction_ids)} transactions.",
        "count": len(transaction_ids),
        "transaction_ids": transaction_ids,
        "alert_ids": alert_ids,
    }


async def _query_transactions(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    date_from = date.fromisoformat(args["date_from"])
    date_to = date.fromisoformat(args["date_to"])
    category_id: uuid.UUID | None = None
    if args.get("category"):
        category = await _resolve_category(team_id, args["category"], db)
        if category is not None:
            category_id = category.id
    account_id: uuid.UUID | None = None
    if args.get("account_name"):
        account = await _resolve_account(team_id, args["account_name"], db)
        if account is not None:
            account_id = account.id

    rows = await finance_service.list_transactions(
        team_id=team_id,
        date_from=date_from,
        date_to=date_to,
        db=db,
        category_id=category_id,
        account_id=account_id,
    )
    direction_filter = args.get("direction")
    if direction_filter in {"income", "expense"}:
        rows = [row for row in rows if row.direction == direction_filter]
    limit = int(args.get("limit") or 20)
    rows = rows[:limit]
    items = [
        {
            "id": str(row.id),
            "amount_fen": row.amount_fen,
            "direction": row.direction,
            "description": row.description,
            "transaction_date": row.transaction_date.isoformat(),
            "category_id": str(row.category_id) if row.category_id else None,
            "account_id": str(row.account_id),
        }
        for row in rows
    ]
    total_fen = sum(item["amount_fen"] for item in items)
    return {
        "status": "queried",
        "message": f"Found {len(items)} transactions, total {total_fen} fen.",
        "count": len(items),
        "total_fen": total_fen,
        "items": items,
    }


async def _analyze_spending(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    period_start = date.fromisoformat(args["period_start"])
    period_end = date.fromisoformat(args["period_end"])
    group_by = args["group_by"]
    current_rows = await _aggregate_expense(team_id, period_start, period_end, group_by, db)
    current_total = sum(row["amount_fen"] for row in current_rows)

    compare_total: int | None = None
    compare_rows: list[dict[str, Any]] | None = None
    delta: int | None = None
    if args.get("compare_with") == "prev_period":
        span_days = (period_end - period_start).days + 1
        prev_end = period_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=span_days - 1)
        compare_rows = await _aggregate_expense(team_id, prev_start, prev_end, group_by, db)
        compare_total = sum(row["amount_fen"] for row in compare_rows)
        delta = current_total - compare_total

    return {
        "status": "analyzed",
        "message": f"Spending {current_total} fen from {period_start} to {period_end}.",
        "group_by": group_by,
        "current_total_fen": current_total,
        "current_rows": current_rows,
        "compare_total_fen": compare_total,
        "compare_rows": compare_rows,
        "delta_fen": delta,
    }


async def _check_budget(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    category_id: uuid.UUID | None = None
    if args.get("category"):
        category = await _resolve_category(team_id, args["category"], db)
        if category is not None:
            category_id = category.id

    stmt = select(Budget).where(Budget.team_id == team_id, Budget.is_active == True)
    if category_id is not None:
        stmt = stmt.where(Budget.category_id == category_id)
    result = await db.execute(stmt)
    budgets = list(result.scalars().all())

    items: list[dict[str, Any]] = []
    for budget in budgets:
        usage = await finance_service.get_budget_usage(budget.id, team_id, db)
        items.append(
            {
                "budget_id": str(budget.id),
                "category_id": str(budget.category_id) if budget.category_id else None,
                "amount_fen": int(usage["amount_fen"]),
                "spent_fen": int(usage["spent_fen"]),
                "usage_ratio": float(usage["usage_ratio"]),
                "period": usage["period"],
            }
        )
    return {
        "status": "checked",
        "message": f"Found {len(items)} active budget(s).",
        "budgets": items,
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


async def _aggregate_expense(
    team_id: uuid.UUID,
    period_start: date,
    period_end: date,
    group_by: str,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    if group_by == "category":
        group_column = Transaction.category_id
    elif group_by == "account":
        group_column = Transaction.account_id
    elif group_by == "week":
        group_column = func.date_trunc("week", Transaction.transaction_date)
    else:
        group_column = Transaction.transaction_date
    result = await db.execute(
        select(
            group_column.label("group_key"),
            func.coalesce(func.sum(Transaction.amount_fen), 0).label("amount_fen"),
        )
        .where(
            and_(
                Transaction.team_id == team_id,
                Transaction.direction == "expense",
                Transaction.transaction_date >= period_start,
                Transaction.transaction_date <= period_end,
                Transaction.deleted_at.is_(None),
            )
        )
        .group_by(group_column)
        .order_by(group_column)
    )
    rows: list[dict[str, Any]] = []
    for group_key, amount_fen in result.all():
        if isinstance(group_key, date):
            key = group_key.isoformat()
        elif group_key is None:
            key = None
        else:
            key = str(group_key)
        rows.append({"group_key": key, "amount_fen": int(amount_fen)})
    return rows
