import hashlib
import json
from typing import Any

from app.cache.normalize import make_llm_cache_key, normalize_query

_TTL_SECONDS = 3600


def _team_index_value(team_id: str, year_month: str) -> str:
    raw = f"{team_id}:{year_month}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def get_cached_response(
    redis: Any, query: str, team_id: str, year_month: str
) -> dict[str, Any] | None:
    key = make_llm_cache_key(query, team_id, year_month)
    raw = await redis.get(key)
    if raw is None:
        return None
    decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return json.loads(decoded)


async def set_cached_response(
    redis: Any,
    query: str,
    team_id: str,
    year_month: str,
    payload: dict[str, Any],
) -> None:
    if not normalize_query(query):
        return
    key = make_llm_cache_key(query, team_id, year_month)
    await redis.setex(key, _TTL_SECONDS, json.dumps(payload, ensure_ascii=False))
    await redis.setex(f"{key}:idx", _TTL_SECONDS, _team_index_value(team_id, year_month))


async def invalidate_team_cache(redis: Any, team_id: str, year_month: str) -> int:
    target = _team_index_value(team_id, year_month)
    to_delete: list[str] = []
    async for idx_key in redis.scan_iter(match="llm:resp:*:idx"):
        key_str = idx_key.decode("utf-8") if isinstance(idx_key, (bytes, bytearray)) else idx_key
        value = await redis.get(key_str)
        if value is None:
            continue
        value_str = value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
        if value_str == target:
            to_delete.append(key_str.removesuffix(":idx"))
            to_delete.append(key_str)
    if not to_delete:
        return 0
    await redis.delete(*to_delete)
    return len(to_delete) // 2
