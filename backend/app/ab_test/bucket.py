import hashlib
from typing import Any

from app.ab_test.registry import get_active_experiment


def deterministic_bucket(team_id: str) -> int:
    digest = hashlib.md5(team_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 100


async def assign_prompt_version(
    team_id: str,
    redis: Any,
    default: str,
) -> str:
    cfg = await get_active_experiment(redis)
    if not cfg:
        return default

    bucket = deterministic_bucket(team_id)
    if bucket < int(cfg.get("traffic_pct", 0)):
        return str(cfg["candidate"])
    return str(cfg["baseline"])
