import json
import uuid
from datetime import date
from typing import Any

_TTL_SECONDS = 300


def budget_summary_key(team_id: uuid.UUID, category_id: uuid.UUID, year_month: str) -> str:
    return f"budget:summary:{team_id}:{category_id}:{year_month}"


def current_year_month() -> str:
    today = date.today()
    return f"{today.year:04d}-{today.month:02d}"


async def get_cached_budget_summary(
    redis: Any,
    team_id: uuid.UUID,
    category_id: uuid.UUID,
    year_month: str,
) -> dict[str, Any] | None:
    raw = await redis.get(budget_summary_key(team_id, category_id, year_month))
    if raw is None:
        return None
    decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return json.loads(decoded)


async def set_cached_budget_summary(
    redis: Any,
    team_id: uuid.UUID,
    category_id: uuid.UUID,
    year_month: str,
    payload: dict[str, Any],
) -> None:
    await redis.setex(
        budget_summary_key(team_id, category_id, year_month),
        _TTL_SECONDS,
        json.dumps(payload, ensure_ascii=False, default=str),
    )


async def invalidate_budget_summary(
    redis: Any,
    team_id: uuid.UUID,
    category_id: uuid.UUID,
    year_month: str,
) -> None:
    await redis.delete(budget_summary_key(team_id, category_id, year_month))
