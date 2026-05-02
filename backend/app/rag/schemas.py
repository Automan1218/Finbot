import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    team_id: uuid.UUID
    title: str = Field(min_length=1, max_length=200)
    source_type: str = Field(min_length=1, max_length=50)
    text: str = Field(min_length=1)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    team_id: uuid.UUID
    title: str
    source_type: str
    created_by: uuid.UUID
    created_at: datetime
