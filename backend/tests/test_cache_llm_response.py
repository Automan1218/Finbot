import fnmatch
import json
import uuid

import pytest

from app.cache.llm_response import (
    get_cached_response,
    invalidate_team_cache,
    set_cached_response,
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

    async def scan_iter(self, match: str):
        for key in list(self.store.keys()):
            if fnmatch.fnmatchcase(key, match):
                yield key

    async def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                self.ttls.pop(k, None)
                removed += 1
        return removed


@pytest.mark.asyncio
async def test_set_and_get_roundtrip():
    redis = FakeRedis()
    team_id = str(uuid.uuid4())
    payload = {"answer": "餐饮 1500 fen", "tool": "analyze"}

    await set_cached_response(redis, "查餐饮", team_id, "2026-05", payload)
    cached = await get_cached_response(redis, "查餐饮", team_id, "2026-05")

    assert cached == payload


@pytest.mark.asyncio
async def test_get_cache_miss_returns_none():
    redis = FakeRedis()
    cached = await get_cached_response(redis, "未命中", "team-x", "2026-05")
    assert cached is None


@pytest.mark.asyncio
async def test_setex_ttl_is_3600():
    redis = FakeRedis()
    await set_cached_response(redis, "查餐饮", "team-1", "2026-05", {"x": 1})
    assert 3600 in redis.ttls.values()


@pytest.mark.asyncio
async def test_invalidate_team_cache_deletes_only_team_keys():
    redis = FakeRedis()
    team_a = str(uuid.uuid4())
    team_b = str(uuid.uuid4())

    await set_cached_response(redis, "餐饮", team_a, "2026-05", {"a": 1})
    await set_cached_response(redis, "交通", team_a, "2026-05", {"a": 2})
    await set_cached_response(redis, "餐饮", team_b, "2026-05", {"b": 1})

    deleted = await invalidate_team_cache(redis, team_a, "2026-05")

    assert deleted == 2
    assert await get_cached_response(redis, "餐饮", team_a, "2026-05") is None
    assert await get_cached_response(redis, "餐饮", team_b, "2026-05") == {"b": 1}
