import uuid
from datetime import date
from typing import Literal

from pydantic import BaseModel, field_validator


class AccountCreate(BaseModel):
    team_id: uuid.UUID
    name: str
    type: Literal["bank", "credit", "cash", "investment"]
    currency: str = "CNY"
    balance_fen: int = 0


class AccountUpdate(BaseModel):
    name: str | None = None
    type: Literal["bank", "credit", "cash", "investment"] | None = None
    balance_fen: int | None = None
    is_active: bool | None = None


class AccountResponse(BaseModel):
    id: uuid.UUID
    team_id: uuid.UUID
    name: str
    type: str
    currency: str
    balance_fen: int
    is_active: bool

    model_config = {"from_attributes": True}


class CategoryCreate(BaseModel):
    team_id: uuid.UUID
    name: str
    icon: str | None = None
    parent_id: uuid.UUID | None = None


class CategoryResponse(BaseModel):
    id: uuid.UUID
    team_id: uuid.UUID | None
    name: str
    icon: str | None
    parent_id: uuid.UUID | None

    model_config = {"from_attributes": True}


class TransactionCreate(BaseModel):
    team_id: uuid.UUID
    account_id: uuid.UUID
    category_id: uuid.UUID | None = None
    amount_fen: int
    direction: Literal["income", "expense"]
    description: str | None = None
    transaction_date: date

    @field_validator("amount_fen")
    @classmethod
    def amount_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("amount_fen must be positive")
        return v


class TransactionUpdate(BaseModel):
    category_id: uuid.UUID | None = None
    amount_fen: int | None = None
    direction: Literal["income", "expense"] | None = None
    description: str | None = None
    transaction_date: date | None = None


class TransactionResponse(BaseModel):
    id: uuid.UUID
    team_id: uuid.UUID
    account_id: uuid.UUID
    category_id: uuid.UUID | None
    amount_fen: int
    direction: str
    description: str | None
    transaction_date: date
    created_by: uuid.UUID
    created_by_ai: bool

    model_config = {"from_attributes": True}


class BudgetCreate(BaseModel):
    team_id: uuid.UUID
    category_id: uuid.UUID
    amount_fen: int
    period: Literal["monthly", "quarterly"]
    alert_threshold: float = 0.8


class BudgetResponse(BaseModel):
    id: uuid.UUID
    team_id: uuid.UUID
    category_id: uuid.UUID
    amount_fen: int
    period: str
    alert_threshold: float
    is_active: bool

    model_config = {"from_attributes": True}


class BudgetUsage(BaseModel):
    budget_id: uuid.UUID
    amount_fen: int
    spent_fen: int
    usage_ratio: float
    period: str
    period_start: date
    period_end: date
