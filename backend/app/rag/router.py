import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.rag import schemas, service
from app.rag.ingest import ingest_document
from app.teams.service import get_member_role

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

_ADMIN_ROLES = ("owner", "admin")
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


async def _require_admin(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin or owner required")
    return team_id, current_user


@router.post("/documents", response_model=schemas.DocumentResponse, status_code=201)
async def create_document(
    body: schemas.DocumentCreate,
    ctx: tuple[uuid.UUID, User] = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    team_id, user = ctx
    if body.team_id != team_id:
        raise HTTPException(status_code=400, detail="team_id mismatch")
    try:
        return await ingest_document(
            team_id=team_id,
            user_id=user.id,
            title=body.title,
            source_type=body.source_type,
            text=body.text,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/documents", response_model=list[schemas.DocumentResponse])
async def list_docs(
    ctx: tuple[uuid.UUID, User] = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_documents(team_id, db)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_doc(
    doc_id: uuid.UUID,
    ctx: tuple[uuid.UUID, User] = Depends(_require_admin),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    try:
        await service.delete_document(doc_id, team_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
