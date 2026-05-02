import uuid

import pytest

from app.cache.categories import (
    categories_cache_key,
    get_cached_categories,
    invalidate_categories,
    set_cached_categories,
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

    async def delete(self, *keys: str) -> int:
        removed = 0
        for k in keys:
            if k in self.store:
                del self.store[k]
                self.ttls.pop(k, None)
                removed += 1
        return removed


def test_cache_key_format():
    team = uuid.uuid4()
    assert categories_cache_key(team) == f"categories:{team}"


@pytest.mark.asyncio
async def test_roundtrip():
    redis = FakeRedis()
    team = uuid.uuid4()
    categories = [
        {"id": str(uuid.uuid4()), "name": "餐饮", "icon": None, "parent_id": None, "team_id": str(team)},
        {"id": str(uuid.uuid4()), "name": "交通", "icon": None, "parent_id": None, "team_id": str(team)},
    ]

    await set_cached_categories(redis, team, categories)
    cached = await get_cached_categories(redis, team)

    assert cached == categories


@pytest.mark.asyncio
async def test_ttl_is_3600():
    redis = FakeRedis()
    await set_cached_categories(redis, uuid.uuid4(), [])
    assert list(redis.ttls.values()) == [3600]


@pytest.mark.asyncio
async def test_invalidate_removes_key():
    redis = FakeRedis()
    team = uuid.uuid4()
    await set_cached_categories(redis, team, [{"id": "x", "name": "test"}])
    await invalidate_categories(redis, team)
    assert await get_cached_categories(redis, team) is None


@pytest.mark.asyncio
async def test_create_invalidates_list_cache(db_session, finance_setup):
    from app.cache.categories import categories_cache_key
    from app.core.redis import get_redis as real_get_redis
    from app.finance.service import create_category, list_categories

    _, team, _ = finance_setup
    redis = await real_get_redis()
    key = categories_cache_key(team.id)
    await redis.delete(key)

    first = await list_categories(team.id, db_session)
    assert await redis.get(key) is not None

    await create_category(team.id, "Travel", None, None, db_session)
    assert await redis.get(key) is None

    refreshed = await list_categories(team.id, db_session)
    assert len(refreshed) == len(first) + 1
