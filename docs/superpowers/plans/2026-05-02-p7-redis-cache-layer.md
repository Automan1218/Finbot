# P7 Redis Cache Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Redis multi-layer cache (LLM responses, budget summary, categories) plus session sliding-window + summary compression to cut token cost by ~30% (spec §6) and shrink long-conversation context.

**Architecture:** All cache helpers live under `app/cache/`. Each helper exposes `get_*` / `set_*` / `invalidate_*` functions returning native Python types (dict/list/str). Callers in `finance/service.py` invalidate after writes (write-through). `chat/service.py` switches session reads from raw `lrange` to a compressed window builder. Agent path adds an LLM-response cache check before calling `resolve_intent` and writes the response back on `done`.

**Tech Stack:** Redis (`redis.asyncio`), existing `OPENAI_MODEL` for summarization, FastAPI deps unchanged.

---

## File Structure

**New files (under `backend/app/cache/`):**
- `cache/__init__.py` — package marker
- `cache/normalize.py` — `normalize_query` helper (Layer 1 prerequisite)
- `cache/llm_response.py` — Layer 1 (LLM response, 1h TTL, team+month scoped)
- `cache/budget.py` — Layer 2 (budget summary, 5min TTL, write-through invalidation)
- `cache/categories.py` — Layer 3 (categories list, 1h TTL, invalidate on POST/DELETE)
- `cache/session.py` — Sliding window + summary compression (§6.3)

**New tests:**
- `tests/test_cache_normalize.py`
- `tests/test_cache_llm_response.py`
- `tests/test_cache_budget.py`
- `tests/test_cache_categories.py`
- `tests/test_cache_session.py`

**Modified files:**
- `backend/app/finance/service.py` — invalidate budget cache on `create_transaction`/`update_transaction`/`soft_delete_transaction`; invalidate categories cache on `create_category`; serve `get_budget_usage`/`list_categories` through cache
- `backend/app/finance/router.py` — pass redis dep into list/create/update/usage handlers
- `backend/app/chat/service.py` — replace plain `lrange` history read with `build_session_window`; LLM-cache wrap on `run_local_chat_task`

---

### Task 1: Query normalization

**Files:**
- Create: `backend/app/cache/__init__.py`
- Create: `backend/app/cache/normalize.py`
- Create: `backend/tests/test_cache_normalize.py`

- [ ] **Step 1: Create cache package**

Create `backend/app/cache/__init__.py` with empty content.

- [ ] **Step 2: Write failing test**

Create `backend/tests/test_cache_normalize.py`:

```python
from app.cache.normalize import make_llm_cache_key, normalize_query


def test_normalize_strips_punctuation_case_filler():
    assert normalize_query("帮我查餐饮！") == "查餐饮"
    assert normalize_query("请告诉我餐饮花了多少") == "餐饮花了多少"
    assert normalize_query("  Hello,  World!  ") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_query("a   b\n c\t d") == "a b c d"


def test_normalize_returns_empty_for_only_filler():
    assert normalize_query("帮我请一下") == ""


def test_make_llm_cache_key_stable_for_synonymous_inputs():
    k1 = make_llm_cache_key("帮我查餐饮", "team-1", "2026-05")
    k2 = make_llm_cache_key("请查一下餐饮", "team-1", "2026-05")
    k3 = make_llm_cache_key("查餐饮", "team-1", "2026-05")
    assert k1 == k2 == k3
    assert k1.startswith("llm:resp:")


def test_make_llm_cache_key_differs_by_team_or_month():
    base = make_llm_cache_key("查餐饮", "team-1", "2026-05")
    assert base != make_llm_cache_key("查餐饮", "team-2", "2026-05")
    assert base != make_llm_cache_key("查餐饮", "team-1", "2026-06")
```

- [ ] **Step 3: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_normalize.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement normalize_query + cache key builder**

Create `backend/app/cache/normalize.py`:

```python
import hashlib
import re

_FILLERS = ("帮我", "请", "麻烦", "能不能", "可以", "一下", "告诉我")
_PUNCTUATION_RE = re.compile(r"[，。！？、,.!?]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = _PUNCTUATION_RE.sub(" ", cleaned)
    for filler in _FILLERS:
        cleaned = cleaned.replace(filler, "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def make_llm_cache_key(query: str, team_id: str, year_month: str) -> str:
    normalized = normalize_query(query)
    raw = f"{team_id}:{year_month}:{normalized}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"llm:resp:{digest}"
```

- [ ] **Step 5: Run test to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_normalize.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```
git add backend/app/cache/__init__.py backend/app/cache/normalize.py backend/tests/test_cache_normalize.py
git commit -m "feat: P7 query normalization + LLM cache key builder"
```

---

### Task 2: LLM response cache (Layer 1)

**Files:**
- Create: `backend/app/cache/llm_response.py`
- Create: `backend/tests/test_cache_llm_response.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_cache_llm_response.py`:

```python
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
        prefix = match.replace("*", "")
        for key in list(self.store.keys()):
            if key.startswith(prefix):
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
    ttl_values = list(redis.ttls.values())
    assert ttl_values == [3600]


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
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_llm_response.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement LLM response cache**

Create `backend/app/cache/llm_response.py`:

```python
import hashlib
import json
from typing import Any

from app.cache.normalize import make_llm_cache_key, normalize_query

_TTL_SECONDS = 3600


def _team_index_key(team_id: str, year_month: str) -> str:
    raw = f"{team_id}:{year_month}".encode("utf-8")
    return f"llm:resp:idx:{hashlib.sha256(raw).hexdigest()}"


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
    await redis.setex(f"{key}:idx", _TTL_SECONDS, _team_index_key(team_id, year_month))


async def invalidate_team_cache(redis: Any, team_id: str, year_month: str) -> int:
    target = _team_index_key(team_id, year_month)
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
```

- [ ] **Step 4: Run test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_llm_response.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/cache/llm_response.py backend/tests/test_cache_llm_response.py
git commit -m "feat: P7 LLM response cache with team-scoped invalidation"
```

---

### Task 3: Budget summary cache (Layer 2) + write-through

**Files:**
- Create: `backend/app/cache/budget.py`
- Create: `backend/tests/test_cache_budget.py`
- Modify: `backend/app/finance/service.py`

- [ ] **Step 1: Write failing test for cache helpers**

Create `backend/tests/test_cache_budget.py`:

```python
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
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_budget.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement budget cache helpers**

Create `backend/app/cache/budget.py`:

```python
import json
import uuid
from typing import Any

_TTL_SECONDS = 300


def budget_summary_key(team_id: uuid.UUID, category_id: uuid.UUID, year_month: str) -> str:
    return f"budget:summary:{team_id}:{category_id}:{year_month}"


def current_year_month() -> str:
    from datetime import date

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
```

- [ ] **Step 4: Run cache helper tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_budget.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Wire write-through invalidation in finance service**

Edit `backend/app/finance/service.py`. Add import at top:

```python
from app.cache.budget import current_year_month, invalidate_budget_summary
from app.core.redis import get_redis
```

After the body of `create_transaction`, before the final `return tx`, insert:

```python
    if tx.category_id is not None:
        redis = await get_redis()
        await invalidate_budget_summary(redis, tx.team_id, tx.category_id, current_year_month())
```

Apply the same insertion at the end of `update_transaction` (before final `return tx`) and at the end of `soft_delete_transaction` (after `await db.commit()`).

In `get_budget_usage`, add cache read at the top (after `result.scalar_one_or_none()` lookup) and cache write before `return`. Replace the existing function body so it returns from cache when fresh:

```python
async def get_budget_usage(
    budget_id: uuid.UUID, team_id: uuid.UUID, db: AsyncSession
) -> dict:
    from datetime import date as date_type
    from app.cache.budget import (
        current_year_month,
        get_cached_budget_summary,
        set_cached_budget_summary,
    )
    from app.core.redis import get_redis

    result = await db.execute(
        select(Budget).where(Budget.id == budget_id, Budget.team_id == team_id)
    )
    budget = result.scalar_one_or_none()
    if not budget:
        raise ValueError("Budget not found")

    year_month = current_year_month()
    redis = await get_redis()
    cached = await get_cached_budget_summary(redis, team_id, budget.category_id, year_month)
    if cached is not None:
        cached["budget_id"] = budget_id
        return cached

    today = date_type.today()
    if budget.period == "monthly":
        period_start = today.replace(day=1)
        if today.month == 12:
            period_end = date_type(today.year + 1, 1, 1)
        else:
            period_end = date_type(today.year, today.month + 1, 1)
    else:
        quarter = (today.month - 1) // 3
        period_start = date_type(today.year, quarter * 3 + 1, 1)
        if quarter == 3:
            period_end = date_type(today.year + 1, 1, 1)
        else:
            period_end = date_type(today.year, (quarter + 1) * 3 + 1, 1)

    spent_result = await db.execute(
        select(func.coalesce(func.sum(Transaction.amount_fen), 0)).where(
            and_(
                Transaction.team_id == team_id,
                Transaction.category_id == budget.category_id,
                Transaction.direction == "expense",
                Transaction.transaction_date >= period_start,
                Transaction.transaction_date < period_end,
                Transaction.deleted_at.is_(None),
            )
        )
    )
    spent_fen = int(spent_result.scalar() or 0)
    payload = {
        "budget_id": budget_id,
        "amount_fen": budget.amount_fen,
        "spent_fen": spent_fen,
        "usage_ratio": spent_fen / budget.amount_fen if budget.amount_fen > 0 else 0.0,
        "period": budget.period,
        "period_start": period_start,
        "period_end": period_end,
    }
    await set_cached_budget_summary(redis, team_id, budget.category_id, year_month, payload)
    return payload
```

- [ ] **Step 6: Add integration test (write-through)**

Append to `backend/tests/test_cache_budget.py`:

```python
import pytest_asyncio
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


@pytest.mark.asyncio
async def test_write_through_invalidates_cache(db_session, finance_setup):
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
```

- [ ] **Step 7: Run all budget cache tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_budget.py backend/tests/test_finance.py backend/tests/test_finance_service.py -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```
git add backend/app/cache/budget.py backend/app/finance/service.py backend/tests/test_cache_budget.py
git commit -m "feat: P7 budget summary cache with write-through invalidation"
```

---

### Task 4: Categories list cache (Layer 3)

**Files:**
- Create: `backend/app/cache/categories.py`
- Create: `backend/tests/test_cache_categories.py`
- Modify: `backend/app/finance/service.py`

- [ ] **Step 1: Write failing test for cache helpers**

Create `backend/tests/test_cache_categories.py`:

```python
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
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_categories.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement categories cache**

Create `backend/app/cache/categories.py`:

```python
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
```

- [ ] **Step 4: Wire into finance service**

Edit `backend/app/finance/service.py`. Add import:

```python
from app.cache.categories import (
    get_cached_categories,
    invalidate_categories,
    set_cached_categories,
)
```

Replace `create_category` and `list_categories` with cache-aware versions:

```python
async def create_category(
    team_id: uuid.UUID, name: str, icon: str | None, parent_id: uuid.UUID | None, db: AsyncSession
) -> Category:
    cat = Category(team_id=team_id, name=name, icon=icon, parent_id=parent_id)
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    redis = await get_redis()
    await invalidate_categories(redis, team_id)
    return cat


async def list_categories(team_id: uuid.UUID, db: AsyncSession) -> list[Category]:
    redis = await get_redis()
    cached = await get_cached_categories(redis, team_id)
    if cached is not None:
        result_objs = []
        for row in cached:
            cat = Category(
                id=uuid.UUID(row["id"]) if isinstance(row.get("id"), str) else row.get("id"),
                team_id=uuid.UUID(row["team_id"]) if row.get("team_id") else None,
                name=row.get("name"),
                icon=row.get("icon"),
                parent_id=(uuid.UUID(row["parent_id"]) if row.get("parent_id") else None),
            )
            result_objs.append(cat)
        return result_objs

    result = await db.execute(
        select(Category).where(
            or_(Category.team_id == team_id, Category.team_id.is_(None))
        )
    )
    items = list(result.scalars().all())
    payload = [
        {
            "id": str(c.id),
            "team_id": str(c.team_id) if c.team_id else None,
            "name": c.name,
            "icon": c.icon,
            "parent_id": str(c.parent_id) if c.parent_id else None,
        }
        for c in items
    ]
    await set_cached_categories(redis, team_id, payload)
    return items
```

- [ ] **Step 5: Add integration test (write-through)**

Append to `backend/tests/test_cache_categories.py`:

```python
from app.cache.categories import categories_cache_key
from app.core.redis import get_redis as real_get_redis
from app.finance.service import create_category, list_categories


@pytest.mark.asyncio
async def test_create_invalidates_list_cache(db_session, finance_setup):
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
```

- [ ] **Step 6: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_categories.py backend/tests/test_finance.py backend/tests/test_finance_service.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```
git add backend/app/cache/categories.py backend/app/finance/service.py backend/tests/test_cache_categories.py
git commit -m "feat: P7 categories list cache with write-through invalidation"
```

---

### Task 5: Session sliding window + summary compression

**Files:**
- Create: `backend/app/cache/session.py`
- Create: `backend/tests/test_cache_session.py`
- Modify: `backend/app/chat/service.py`

- [ ] **Step 1: Write failing test for window logic**

Create `backend/tests/test_cache_session.py`:

```python
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.cache.session import (
    SESSION_TTL_SECONDS,
    build_session_window,
    summarize_messages,
)


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[bytes]] = {}
        self.ttls: dict[str, int] = {}

    async def lrange(self, key: str, start: int, stop: int):
        items = self.lists.get(key, [])
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    async def rpush(self, key: str, value):
        self.lists.setdefault(key, []).append(value if isinstance(value, bytes) else value.encode("utf-8"))

    async def delete(self, key: str):
        self.lists.pop(key, None)
        self.ttls.pop(key, None)

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl


def _build_client(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _push_messages(redis: FakeRedis, key: str, count: int) -> None:
    for i in range(count):
        msg = {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        redis.lists.setdefault(key, []).append(json.dumps(msg).encode("utf-8"))


@pytest.mark.asyncio
async def test_short_history_returns_messages_unchanged():
    redis = FakeRedis()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    key = f"session:{user_id}:{conv_id}"
    _push_messages(redis, key, 5)

    window = await build_session_window(redis, user_id, conv_id)

    assert len(window) == 5
    assert window[0]["content"] == "m0"


@pytest.mark.asyncio
async def test_long_history_returns_compressed_window(monkeypatch):
    redis = FakeRedis()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    key = f"session:{user_id}:{conv_id}"
    _push_messages(redis, key, 25)

    fake_summarize = AsyncMock(return_value="summary text")
    monkeypatch.setattr("app.cache.session.summarize_messages", fake_summarize)

    window = await build_session_window(
        redis, user_id, conv_id, max_history=10, compress_at=20
    )

    assert window[0]["role"] == "system"
    assert "summary text" in window[0]["content"]
    assert len(window) == 11  # 1 summary + 10 most recent
    assert window[-1]["content"] == "m24"
    assert redis.ttls[key] == SESSION_TTL_SECONDS


@pytest.mark.asyncio
async def test_summarize_messages_calls_llm_with_history():
    client = _build_client("compressed summary")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    summary = await summarize_messages(messages, client=client)

    assert summary == "compressed summary"
    client.chat.completions.create.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_session.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement session window**

Create `backend/app/cache/session.py`:

```python
import json
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings

SESSION_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_HISTORY = 10
DEFAULT_COMPRESS_AT = 20

_SUMMARIZE_PROMPT = (
    "概括以下对话历史的关键信息，保留用户意图、已确认事实、未解决问题。"
    "限制在 200 字以内，使用第三人称描述：\n\n{transcript}"
)


def _session_key(user_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
    return f"session:{user_id}:{conversation_id}"


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


async def summarize_messages(
    messages: list[dict[str, Any]],
    client: AsyncOpenAI | None = None,
) -> str:
    if not messages:
        return ""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "user", "content": _SUMMARIZE_PROMPT.format(transcript=transcript)}
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def build_session_window(
    redis: Any,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    max_history: int = DEFAULT_MAX_HISTORY,
    compress_at: int = DEFAULT_COMPRESS_AT,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    key = _session_key(user_id, conversation_id)
    raw_messages = await redis.lrange(key, 0, -1)
    messages = [json.loads(_decode(item)) for item in raw_messages]

    if len(messages) <= compress_at:
        return messages[-max_history:] if len(messages) > max_history else messages

    older = messages[:-max_history]
    summary_text = await summarize_messages(older, client=client)
    summary_msg = {"role": "system", "content": f"历史摘要: {summary_text}"}
    recent = messages[-max_history:]

    await redis.delete(key)
    for msg in [summary_msg, *recent]:
        await redis.rpush(key, json.dumps(msg, ensure_ascii=False))
    await redis.expire(key, SESSION_TTL_SECONDS)
    return [summary_msg, *recent]
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_cache_session.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Wire into chat service**

Edit `backend/app/chat/service.py`. Add import:

```python
from app.cache.session import build_session_window
```

Add a new function below `get_history`:

```python
async def get_compressed_window(
    redis: Redis,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> list[dict[str, Any]]:
    return await build_session_window(redis, user_id, conversation_id)
```

(`Redis` is already imported via `from redis.asyncio import Redis` in router; if not present in service, add it.)

- [ ] **Step 6: Commit**

```
git add backend/app/cache/session.py backend/app/chat/service.py backend/tests/test_cache_session.py
git commit -m "feat: P7 session sliding window with summary compression"
```

---

### Task 6: LLM response cache wiring in agent task

**Files:**
- Modify: `backend/app/chat/service.py`
- Modify: `backend/tests/test_chat_service.py`

- [ ] **Step 1: Write failing test**

Append to `backend/tests/test_chat_service.py`:

```python
import json
import uuid
from unittest.mock import AsyncMock

import pytest

from app.cache.normalize import make_llm_cache_key
from app.chat.service import run_local_chat_task


@pytest.mark.asyncio
async def test_run_local_chat_task_uses_cached_llm_response(monkeypatch):
    fake_redis = AsyncMock()
    cached_payload = {
        "intent": {"name": "clarify", "arguments": {"question": "cached"}},
        "execution": None,
        "response": "cached answer",
    }

    async def fake_get(key):
        if key.startswith("llm:resp:"):
            return json.dumps(cached_payload).encode("utf-8")
        return None

    fake_redis.get = AsyncMock(side_effect=fake_get)
    fake_redis.rpush = AsyncMock()
    fake_redis.expire = AsyncMock()
    fake_redis.set = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.publish = AsyncMock()

    fake_resolve = AsyncMock()
    monkeypatch.setattr("app.chat.service.resolve_intent", fake_resolve)

    task_id = uuid.uuid4()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    team_id = uuid.uuid4()

    await run_local_chat_task(fake_redis, task_id, user_id, conv_id, "查餐饮", team_id)

    fake_resolve.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_chat_service.py::test_run_local_chat_task_uses_cached_llm_response -v
```
Expected: FAIL (cache check not yet wired)

- [ ] **Step 3: Wire LLM cache into run_local_chat_task**

Edit `backend/app/chat/service.py`. Add imports:

```python
from app.cache.llm_response import get_cached_response, set_cached_response
from app.cache.budget import current_year_month
```

Replace `run_local_chat_task` body. The new flow checks cache before agent steps and writes back on success:

```python
async def run_local_chat_task(
    redis: Redis,
    task_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    team_id: uuid.UUID | None = None,
) -> None:
    try:
        year_month = current_year_month()
        if team_id is not None:
            cached = await get_cached_response(redis, message, str(team_id), year_month)
            if cached is not None:
                await append_message(redis, user_id, conversation_id, "assistant", cached["response"])
                await push_task_event(
                    redis,
                    task_id,
                    conversation_id,
                    step="complete",
                    status="done",
                    message=cached["response"],
                    data={"intent": cached.get("intent"), "execution": cached.get("execution"), "cache": "hit"},
                )
                return

        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="agent",
            status="running",
            message="Parsing request",
        )
        intent, intent_source = await resolve_intent(message)
        execution = None
        if team_id is not None:
            await push_task_event(
                redis,
                task_id,
                conversation_id,
                step="execute",
                status="running",
                message="Executing intent",
                data={"intent": intent, "intent_source": intent_source},
            )
            async with get_db_session() as db:
                execution = await execute_intent(intent, team_id, user_id, db)
        response = _build_agent_response(intent, execution)
        await append_message(redis, user_id, conversation_id, "assistant", response)
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="complete",
            status="done",
            message=response,
            data={"intent": intent, "intent_source": intent_source, "execution": execution},
        )
        if team_id is not None and intent.get("name") != "clarify":
            await set_cached_response(
                redis,
                message,
                str(team_id),
                year_month,
                {"intent": intent, "execution": execution, "response": response},
            )
    except Exception as exc:
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="error",
            status="error",
            message=str(exc),
        )
```

- [ ] **Step 4: Run test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_chat_service.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add backend/app/chat/service.py backend/tests/test_chat_service.py
git commit -m "feat: P7 wire LLM response cache into agent task"
```

---

### Task 7: Full-suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run full test suite**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/ -q
```
Expected: all green; new tests count >= old + 16.

- [ ] **Step 2: If failures, fix and re-run**

Common follow-ups:
- `current_year_month` mismatch in tests crossing month boundary — pin to fixed string in tests.
- Cache invalidation not firing because `tx.team_id` returns SQLAlchemy lazy-loaded UUID — call `await db.refresh(tx)` first.
- `list_categories` test expecting ORM Category but cache returns reconstructed — assert by name/id, not identity.

- [ ] **Step 3: Final commit (if any fix needed)**

```
git add -A
git commit -m "fix: P7 follow-up adjustments after full-suite run"
```

If suite passes cleanly first time, no commit needed — milestone done.

---

## Self-Review

**Spec coverage:**
- §6.1 Query 标准化 → Task 1 (`normalize_query`, `make_llm_cache_key`)
- §6.2 Layer 1 LLM 响应缓存 → Task 2 + Task 6
- §6.2 Layer 2 预算聚合缓存 → Task 3
- §6.2 Layer 3 分类列表缓存 → Task 4
- §6.2 Layer 4 Embedding 缓存 → P6 (already done)
- §6.2 Layer 5 Query Rewrite 缓存 → P6 (already done)
- §6.3 Session 滑动窗口 + 历史压缩 → Task 5

**Out of scope (handled in P8+):**
- §7 TTFT optimization (parallel context, prefix caching, streaming) — P8
- §8 Eval / RAGAS / A/B — P9
- LLM cache invalidation on transaction write (Layer 1 says "同 team 新增 transaction → DEL") — Task 3 invalidates budget summary; Layer 1 invalidation covered through team-scoped key (`invalidate_team_cache`) but caller hookup deferred (low ROI for P7; explicit invalidation only fires when needed via direct call).

**Placeholder scan:** none.

**Type consistency:**
- `make_llm_cache_key(query, team_id, year_month)` signature consistent across Tasks 1, 2, 6.
- `current_year_month()` defined in Task 3, reused in Task 6 — same source.
- `budget_summary_key(team_id, category_id, year_month)` matches usage in `get_budget_usage` and write-through hooks.
- `build_session_window` returns `list[dict[str, Any]]`; chat service consumer matches.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-02-p7-redis-cache-layer.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — same session, batched checkpoints.

Which approach?
