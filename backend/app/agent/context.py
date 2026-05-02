import asyncio
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.session import build_session_window
from app.rag.retrieve import smart_retrieve


async def load_context_parallel(
    redis: Any,
    db: AsyncSession | None,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    team_id: uuid.UUID | None,
    query: str,
) -> dict[str, Any]:
    session_task = build_session_window(redis, user_id, conversation_id)
    skip_rag = team_id is None or db is None or not query.strip()

    if skip_rag:
        history = await session_task
        return {
            "history": history,
            "rag": [],
            "team_id": team_id,
            "user_id": user_id,
        }

    rag_task = smart_retrieve(query, team_id, db)
    history, rag = await asyncio.gather(session_task, rag_task)
    return {
        "history": history,
        "rag": rag,
        "team_id": team_id,
        "user_id": user_id,
    }
