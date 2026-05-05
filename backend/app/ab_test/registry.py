import json
from typing import Any

EXPERIMENT_KEY = "experiment:active"
_TTL_SECONDS = 7 * 24 * 60 * 60


async def get_active_experiment(redis: Any) -> dict[str, Any] | None:
    raw = await redis.get(EXPERIMENT_KEY)
    if raw is None:
        return None
    decoded = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    return json.loads(decoded)


async def set_experiment(
    redis: Any,
    baseline: str,
    candidate: str,
    traffic_pct: int,
) -> None:
    pct = max(0, min(100, int(traffic_pct)))
    payload = {"baseline": baseline, "candidate": candidate, "traffic_pct": pct}
    await redis.setex(EXPERIMENT_KEY, _TTL_SECONDS, json.dumps(payload))


async def clear_experiment(redis: Any) -> None:
    await redis.delete(EXPERIMENT_KEY)
