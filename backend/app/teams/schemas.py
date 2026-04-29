import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


class TeamCreate(BaseModel):
    name: str


class TeamUpdate(BaseModel):
    name: str | None = None


class TeamResponse(BaseModel):
    id: uuid.UUID
    name: str
    owner_id: uuid.UUID
    plan: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MemberCreate(BaseModel):
    email: EmailStr
    role: Literal["admin", "member", "viewer"] = "member"


class MemberUpdate(BaseModel):
    role: Literal["owner", "admin", "member", "viewer"]


class MemberResponse(BaseModel):
    id: uuid.UUID
    team_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    joined_at: datetime

    model_config = {"from_attributes": True}
