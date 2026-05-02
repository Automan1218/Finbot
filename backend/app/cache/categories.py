import json
import uuid
from typing import Any

_TTL_SECONDS = 3600


def categories_cache_key(team_id: uuid.UUID) -> str:
    return f"categories:{team_id}"


async def get_cached_categories(redis: Any, team_id: uuid.UUID) -> list[dict[str, Any]] | None:
    raw = await redis.get(categories_cache_key(team_id))
    if raw is None:
        return None
    decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return json.loads(decoded)


async def set_cached_categories(
    redis: Any, team_id: uuid.UUID, categories: list[dict[str, Any]]
) -> None:
    await redis.setex(
        categories_cache_key(team_id),
        _TTL_SECONDS,
        json.dumps(categories, ensure_ascii=False, default=str),
    )


async def invalidate_categories(redis: Any, team_id: uuid.UUID) -> None:
    await redis.delete(categories_cache_key(team_id))
