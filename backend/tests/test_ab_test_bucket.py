import uuid

import pytest

from app.ab_test.bucket import assign_prompt_version, deterministic_bucket
from app.ab_test.registry import (
    clear_experiment,
    get_active_experiment,
    set_experiment,
)


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value.encode("utf-8")
        self.ttls[key] = ttl

    async def set(self, key: str, value, ex=None):
        self.store[key] = value if isinstance(value, bytes) else value.encode("utf-8")

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.store:
                del self.store[key]
                removed += 1
        return removed


def test_deterministic_bucket_in_range():
    team = uuid.UUID("11111111-1111-1111-1111-111111111111")
    bucket = deterministic_bucket(str(team))
    assert 0 <= bucket < 100


def test_deterministic_bucket_stable():
    team_str = "00000000-0000-0000-0000-000000000001"
    assert deterministic_bucket(team_str) == deterministic_bucket(team_str)


@pytest.mark.asyncio
async def test_set_get_clear_experiment_roundtrip():
    redis = FakeRedis()
    await set_experiment(redis, baseline="v1.0", candidate="v1.1", traffic_pct=20)
    cfg = await get_active_experiment(redis)
    assert cfg == {"baseline": "v1.0", "candidate": "v1.1", "traffic_pct": 20}

    await clear_experiment(redis)
    assert await get_active_experiment(redis) is None


@pytest.mark.asyncio
async def test_assign_returns_baseline_when_no_experiment():
    redis = FakeRedis()
    version = await assign_prompt_version("any-team", redis, default="v1.0")
    assert version == "v1.0"


@pytest.mark.asyncio
async def test_assign_routes_traffic_pct_to_candidate():
    redis = FakeRedis()
    await set_experiment(redis, baseline="v1.0", candidate="v1.1", traffic_pct=100)

    version = await assign_prompt_version("any-team", redis, default="v0")
    assert version == "v1.1"


@pytest.mark.asyncio
async def test_assign_zero_pct_keeps_baseline():
    redis = FakeRedis()
    await set_experiment(redis, baseline="v1.0", candidate="v1.1", traffic_pct=0)
    version = await assign_prompt_version("any-team", redis, default="v0")
    assert version == "v1.0"
