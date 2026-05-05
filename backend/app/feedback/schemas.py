import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    team_id: uuid.UUID
    task_id: uuid.UUID | None = None
    task_type: str | None = None
    rating: int = Field(ge=-1, le=1)
    feedback_type: str = Field(min_length=1, max_length=50)
    original_output: dict[str, Any] | None = None
    corrected_output: dict[str, Any] | None = None
    prompt_version: str | None = None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    team_id: uuid.UUID
    user_id: uuid.UUID
    task_id: uuid.UUID | None
    task_type: str | None
    rating: int
    feedback_type: str
    failure_type: str | None
    original_output: dict[str, Any] | None
    corrected_output: dict[str, Any] | None
    prompt_version: str | None
    created_at: datetime


class ImplicitSignal(BaseModel):
    team_id: uuid.UUID
    user_id: uuid.UUID
    task_id: uuid.UUID | None = None
    task_type: str | None = None
    signal_type: str = Field(min_length=1, max_length=50)
    prompt_version: str | None = None
