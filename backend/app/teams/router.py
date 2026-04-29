import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.teams import schemas, service

router = APIRouter(prefix="/teams", tags=["teams"])


async def _get_role(
    team_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> str:
    role = await service.get_member_role(team_id, current_user.id, db)
    if role is None:
        raise HTTPException(status_code=403, detail="Not a team member")
    return role


@router.post("", response_model=schemas.TeamResponse, status_code=201)
async def create_team(
    body: schemas.TeamCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.create_team(body.name, current_user.id, db)


@router.get("", response_model=list[schemas.TeamResponse])
async def list_teams(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_teams_for_user(current_user.id, db)


@router.get("/{team_id}", response_model=schemas.TeamResponse)
async def get_team(
    team_id: uuid.UUID,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    team = await service.get_team(team_id, db)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


@router.patch("/{team_id}", response_model=schemas.TeamResponse)
async def update_team(
    team_id: uuid.UUID,
    body: schemas.TeamUpdate,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    if role != "owner":
        raise HTTPException(status_code=403, detail="Owner only")
    try:
        return await service.update_team(team_id, body.model_dump(exclude_none=True), db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{team_id}", status_code=204)
async def delete_team(
    team_id: uuid.UUID,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    if role != "owner":
        raise HTTPException(status_code=403, detail="Owner only")
    try:
        await service.delete_team(team_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{team_id}/members", response_model=list[schemas.MemberResponse])
async def list_members(
    team_id: uuid.UUID,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    return await service.list_members(team_id, db)


@router.post("/{team_id}/members", response_model=schemas.MemberResponse, status_code=201)
async def add_member(
    team_id: uuid.UUID,
    body: schemas.MemberCreate,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Admin or owner required")
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        return await service.add_member(team_id, user.id, body.role, db)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Already a member")


@router.patch(
    "/{team_id}/members/{user_id}/role", response_model=schemas.MemberResponse
)
async def update_member_role(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    body: schemas.MemberUpdate,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    if role != "owner":
        raise HTTPException(status_code=403, detail="Owner only")
    try:
        return await service.update_member_role(team_id, user_id, body.role, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{team_id}/members/{user_id}", status_code=204)
async def remove_member(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    role: str = Depends(_get_role),
    db: AsyncSession = Depends(get_db),
):
    if role not in ("owner", "admin"):
        raise HTTPException(status_code=403, detail="Admin or owner required")
    try:
        await service.remove_member(team_id, user_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
