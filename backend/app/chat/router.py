import asyncio
import json
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.chat import schemas, service
from app.core.database import get_db
from app.core.redis import get_redis
from app.models.user import User
from app.teams.service import get_member_role

router = APIRouter(prefix="/chat", tags=["chat"])

_MEMBER_ROLES = ("owner", "admin", "member")


async def get_chat_redis() -> Redis:
    return await get_redis()


async def _require_chat_member(
    team_id: uuid.UUID,
    current_user: User,
    db: AsyncSession,
) -> None:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _MEMBER_ROLES:
        raise HTTPException(status_code=403, detail="Member or above required")


@router.post("/message", response_model=schemas.ChatTaskResponse, status_code=202)
async def create_message(
    body: schemas.ChatMessageCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_chat_redis),
):
    await _require_chat_member(body.team_id, current_user, db)
    task_id, conversation_id = await service.create_chat_task(
        redis=redis,
        user_id=current_user.id,
        team_id=body.team_id,
        message=body.message,
        conversation_id=body.conversation_id,
    )
    background_tasks.add_task(
        service.run_local_chat_task,
        redis,
        task_id,
        current_user.id,
        conversation_id,
        body.message,
        body.team_id,
    )
    return {
        "task_id": task_id,
        "conversation_id": conversation_id,
        "status": "queued",
    }


@router.get("/stream/{task_id}")
async def stream_task(task_id: uuid.UUID, redis: Redis = Depends(get_chat_redis)):
    async def event_generator():
        sent = 0
        while True:
            events = await service.list_task_events(redis, task_id, sent)
            for event in events:
                sent += 1
                yield {
                    "event": event["status"],
                    "data": json.dumps(event),
                }
                if event["status"] in {"done", "error"}:
                    return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/history", response_model=schemas.ChatHistoryResponse)
async def history(
    conversation_id: uuid.UUID = Query(...),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    redis: Redis = Depends(get_chat_redis),
):
    messages = await service.get_history(
        redis=redis,
        user_id=current_user.id,
        conversation_id=conversation_id,
        page=page,
        size=size,
    )
    return {"conversation_id": conversation_id, "messages": messages}


@router.delete("/history", status_code=204)
async def delete_history(
    conversation_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    redis: Redis = Depends(get_chat_redis),
):
    await service.clear_history(redis, current_user.id, conversation_id)
