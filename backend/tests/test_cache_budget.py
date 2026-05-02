import uuid

import pytest

from app.cache.budget import (
    budget_summary_key,
    get_cached_budget_summary,
    invalidate_budget_summary,
    set_cached_budget_summary,
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


def test_key_format():
    team = uuid.uuid4()
    cat = uuid.uuid4()
    assert budget_summary_key(team, cat, "2026-05") == (
        f"budget:summary:{team}:{cat}:2026-05"
    )


@pytest.mark.asyncio
async def test_set_get_roundtrip():
    redis = FakeRedis()
    team = uuid.uuid4()
    cat = uuid.uuid4()
    payload = {"spent_fen": 1500, "amount_fen": 5000, "ratio": 0.3}

    await set_cached_budget_summary(redis, team, cat, "2026-05", payload)
    cached = await get_cached_budget_summary(redis, team, cat, "2026-05")

    assert cached == payload


@pytest.mark.asyncio
async def test_set_uses_300s_ttl():
    redis = FakeRedis()
    team = uuid.uuid4()
    cat = uuid.uuid4()
    await set_cached_budget_summary(redis, team, cat, "2026-05", {"a": 1})
    assert list(redis.ttls.values()) == [300]


@pytest.mark.asyncio
async def test_invalidate_removes_specific_key():
    redis = FakeRedis()
    team = uuid.uuid4()
    cat = uuid.uuid4()
    await set_cached_budget_summary(redis, team, cat, "2026-05", {"a": 1})
    await invalidate_budget_summary(redis, team, cat, "2026-05")
    assert await get_cached_budget_summary(redis, team, cat, "2026-05") is None


@pytest.mark.asyncio
async def test_write_through_invalidates_cache(db_session, finance_setup):
    from datetime import date as date_type

    from app.cache.budget import budget_summary_key, current_year_month
    from app.core.redis import get_redis as real_get_redis
    from app.finance.service import (
        create_account,
        create_budget,
        create_category,
        create_transaction,
        get_budget_usage,
    )

    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    category = await create_category(team.id, "Food", None, None, db_session)
    budget = await create_budget(team.id, category.id, 10000, "monthly", 0.8, db_session)

    redis = await real_get_redis()
    key = budget_summary_key(team.id, category.id, current_year_month())
    await redis.delete(key)

    first = await get_budget_usage(budget.id, team.id, db_session)
    assert first["spent_fen"] == 0
    assert await redis.get(key) is not None

    await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=category.id,
        amount_fen=2500,
        direction="expense",
        description="lunch",
        transaction_date=date_type.today(),
        created_by=user.id,
        db=db_session,
    )

    assert await redis.get(key) is None
    refreshed = await get_budget_usage(budget.id, team.id, db_session)
    assert refreshed["spent_fen"] == 2500
