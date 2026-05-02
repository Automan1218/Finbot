import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessageCreate(BaseModel):
    team_id: uuid.UUID
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: uuid.UUID | None = None
    stream: bool = False


class ChatTaskResponse(BaseModel):
    task_id: uuid.UUID
    conversation_id: uuid.UUID
    status: Literal["queued"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[ChatMessage]


class ChatTaskEvent(BaseModel):
    task_id: uuid.UUID
    conversation_id: uuid.UUID
    step: str
    status: Literal["queued", "running", "done", "error"]
    message: str
    data: dict[str, Any] | None = None
    created_at: datetime
