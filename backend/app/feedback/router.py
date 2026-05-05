import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.feedback import schemas, service
from app.models.user import User
from app.teams.service import get_member_role

router = APIRouter(prefix="/feedback", tags=["feedback"])

_ANY_ROLES = ("owner", "admin", "member", "viewer")


async def _require_team(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _ANY_ROLES:
        raise HTTPException(status_code=403, detail="Team not found or access denied")
    return team_id, current_user


@router.post("", response_model=schemas.FeedbackResponse, status_code=201)
async def create_feedback(
    body: schemas.FeedbackCreate,
    ctx: tuple[uuid.UUID, User] = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, user = ctx
    if body.team_id != team_id:
        raise HTTPException(status_code=400, detail="team_id mismatch")
    return await service.record_feedback(
        team_id=team_id,
        user_id=user.id,
        task_id=body.task_id,
        task_type=body.task_type,
        rating=body.rating,
        feedback_type=body.feedback_type,
        original_output=body.original_output,
        corrected_output=body.corrected_output,
        prompt_version=body.prompt_version,
        db=db,
    )


@router.get("", response_model=list[schemas.FeedbackResponse])
async def list_feedback(
    ctx: tuple[uuid.UUID, User] = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_feedback(team_id, db)
