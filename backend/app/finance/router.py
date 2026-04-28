import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.finance import schemas, service
from app.models.team import Team
from app.models.user import User

router = APIRouter(prefix="/finance", tags=["finance"])


async def _require_team(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    result = await db.execute(
        select(Team).where(Team.id == team_id, Team.owner_id == current_user.id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Team not found or access denied")
    return team_id, current_user


# --- Accounts ---

@router.post("/accounts", response_model=schemas.AccountResponse, status_code=201)
async def create_account(
    body: schemas.AccountCreate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    if body.team_id != team_id:
        raise HTTPException(status_code=400, detail="team_id mismatch")
    return await service.create_account(
        body.team_id, body.name, body.type, body.currency, body.balance_fen, db
    )


@router.get("/accounts", response_model=list[schemas.AccountResponse])
async def list_accounts(
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_accounts(team_id, db)


@router.patch("/accounts/{account_id}", response_model=schemas.AccountResponse)
async def update_account(
    account_id: uuid.UUID,
    body: schemas.AccountUpdate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        return await service.update_account(
            account_id, team_id, body.model_dump(exclude_none=True), db
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Categories ---

@router.post("/categories", response_model=schemas.CategoryResponse, status_code=201)
async def create_category(
    body: schemas.CategoryCreate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    return await service.create_category(
        body.team_id, body.name, body.icon, body.parent_id, db
    )


@router.get("/categories", response_model=list[schemas.CategoryResponse])
async def list_categories(
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_categories(team_id, db)


# --- Transactions ---

@router.post("/transactions", response_model=schemas.TransactionResponse, status_code=201)
async def create_transaction(
    body: schemas.TransactionCreate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, user = ctx
    return await service.create_transaction(
        team_id=body.team_id,
        account_id=body.account_id,
        category_id=body.category_id,
        amount_fen=body.amount_fen,
        direction=body.direction,
        description=body.description,
        transaction_date=body.transaction_date,
        created_by=user.id,
        db=db,
    )


@router.get("/transactions", response_model=list[schemas.TransactionResponse])
async def list_transactions(
    ctx: tuple = Depends(_require_team),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_transactions(team_id, date_from, date_to, db)


@router.patch("/transactions/{tx_id}", response_model=schemas.TransactionResponse)
async def update_transaction(
    tx_id: uuid.UUID,
    body: schemas.TransactionUpdate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        return await service.update_transaction(
            tx_id, team_id, body.model_dump(exclude_none=True), db
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/transactions/{tx_id}", status_code=204)
async def delete_transaction(
    tx_id: uuid.UUID,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        await service.soft_delete_transaction(tx_id, team_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Budgets ---

@router.post("/budgets", response_model=schemas.BudgetResponse, status_code=201)
async def create_budget(
    body: schemas.BudgetCreate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    return await service.create_budget(
        body.team_id, body.category_id, body.amount_fen, body.period, body.alert_threshold, db
    )


@router.get("/budgets", response_model=list[schemas.BudgetResponse])
async def list_budgets(
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_budgets(team_id, db)


@router.get("/budgets/{budget_id}/usage", response_model=schemas.BudgetUsage)
async def budget_usage(
    budget_id: uuid.UUID,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        return await service.get_budget_usage(budget_id, team_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
