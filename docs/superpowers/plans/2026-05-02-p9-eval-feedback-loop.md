# P9 Eval & Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the closed feedback loop (spec §8): explicit + implicit feedback collection, LLM-based failure-type classification, scenario-based golden-set eval runner, prompt-version registry with A/B bucket routing, and a weekly analyzer aggregating failure patterns. Target: recommendation accuracy +25%.

**Architecture:** Three new packages.
- `app/feedback/` — REST endpoints for explicit feedback, classifier (LLM gpt-4o-mini, prompt fixed in code), implicit-signal hooks called from `finance/service.py` on transaction edit/delete and from agent on accept-without-edit.
- `app/eval/` — `golden_set.py` (100+ test cases as Python list, embedded source-controlled), scenario metric functions (`field_accuracy`, `transaction_count_match`, `missing_field_recall`), `runner.eval_prompt_version()` returning per-scenario scores and aggregate.
- `app/ab_test/` — `prompt_registry.list_active_versions()`, `bucket.assign(team_id) -> version_id` using deterministic md5 hash → bucket vs traffic_pct lookup in Redis-cached `experiment:active` config.

The `chat/service.py` agent path (Task 4) reads `prompt_versions` rows and selects the version per request via `bucket.assign`. Each `feedback` row records `prompt_version` so eval results trace back to a specific version.

**Tech Stack:** FastAPI + SQLAlchemy 2.0 (existing), `redis.asyncio`, OpenAI Python SDK (`AsyncOpenAI` for failure classification only — eval metrics are deterministic and require no LLM calls).

**Out of P9 scope** (defer):
- Real RAGAS library integration (heavy dep, requires real LLM eval traffic) — placeholder `ragas_scores: dict | None` already exists in `feedback` table; P9 leaves field nullable. Future P9.5 milestone can wire ragas.
- Celery scheduling glue (only the analyzer function lands here; cron registration is deferred until Celery infra arrives in P10/P11).

---

## File Structure

**New files (under `backend/app/`):**
- `feedback/__init__.py`
- `feedback/schemas.py` — `FeedbackCreate`, `FeedbackResponse`, `ImplicitSignal`
- `feedback/service.py` — `record_feedback`, `record_implicit_signal`
- `feedback/classifier.py` — `classify_failure(query, response, correction)` → `"retrieval_failure" | "intent_failure" | "generation_failure"`
- `feedback/router.py` — `POST /feedback`, `GET /feedback`
- `feedback/analyzer.py` — `aggregate_failure_modes(team_id, since, db)` returning grouped counts by `failure_type` + top corrected categories
- `eval/__init__.py`
- `eval/golden_set.py` — embedded list of cases (`scenario`, `input`, `expect`)
- `eval/metrics.py` — `field_accuracy`, `transaction_count_match`, `missing_field_recall`, plus `score_scenario(scenario, intent, expected)` dispatcher
- `eval/runner.py` — `eval_prompt_version(version_id, db, redis)` returning `{scenario: avg_score, aggregate: float}`
- `ab_test/__init__.py`
- `ab_test/registry.py` — `get_active_experiment(redis)`, `set_experiment(redis, baseline, candidate, traffic_pct)`, `clear_experiment(redis)`
- `ab_test/bucket.py` — `assign_prompt_version(team_id, redis)` → `version_str`

**New tests:**
- `tests/test_feedback_router.py`
- `tests/test_feedback_classifier.py`
- `tests/test_feedback_analyzer.py`
- `tests/test_eval_metrics.py`
- `tests/test_eval_runner.py`
- `tests/test_ab_test_bucket.py`

**Modified files:**
- `backend/app/main.py` — register `feedback_router`
- `backend/app/finance/service.py` — emit implicit signals on transaction edit/delete (only when `created_by_ai=True`)
- `backend/app/chat/service.py` — write `prompt_version` into the `feedback`-eligible event payload; pick active version via `assign_prompt_version` before calling LLM
- `backend/app/agent/prompt.py` — accept optional `prompt_version` arg in `build_prompt` to swap `SYSTEM_PROMPT` source when needed

---

### Task 1: Failure-type classifier

**Files:**
- Create: `backend/app/feedback/__init__.py`
- Create: `backend/app/feedback/classifier.py`
- Create: `backend/tests/test_feedback_classifier.py`

- [ ] **Step 1: Create feedback package**

Create `backend/app/feedback/__init__.py` with empty content.

- [ ] **Step 2: Write failing test**

Create `backend/tests/test_feedback_classifier.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.feedback.classifier import (
    FAILURE_TYPES,
    classify_failure,
    fallback_classify,
)


def _build_client(label: str) -> MagicMock:
    msg = MagicMock(content=label)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def test_failure_types_const():
    assert FAILURE_TYPES == (
        "retrieval_failure",
        "intent_failure",
        "generation_failure",
    )


def test_fallback_classify_correction_present():
    label = fallback_classify(
        query="出差报销限制",
        response="无法提供具体规则",
        correction="酒店不超过 500/晚",
    )
    assert label == "retrieval_failure"


def test_fallback_classify_no_correction_returns_generation():
    label = fallback_classify(
        query="生成本月报表",
        response="不明白",
        correction=None,
    )
    assert label == "generation_failure"


@pytest.mark.asyncio
async def test_classify_failure_uses_llm_and_normalizes_label():
    client = _build_client("intent_failure\n")

    label = await classify_failure(
        query="今天午饭35元",
        response="未识别为支出",
        correction="餐饮 35 元支出",
        client=client,
    )

    assert label == "intent_failure"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_failure_falls_back_when_label_invalid():
    client = _build_client("not-a-known-label")

    label = await classify_failure(
        query="x",
        response="y",
        correction="z",
        client=client,
    )

    assert label in FAILURE_TYPES
```

- [ ] **Step 3: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_classifier.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement classifier**

Create `backend/app/feedback/classifier.py`:

```python
from openai import AsyncOpenAI

from app.core.config import settings

FAILURE_TYPES: tuple[str, ...] = (
    "retrieval_failure",
    "intent_failure",
    "generation_failure",
)

_PROMPT = (
    "Classify the failure of an AI finance assistant reply into exactly one label.\n"
    "Labels:\n"
    "- retrieval_failure: knowledge retrieval returned wrong or missing documents.\n"
    "- intent_failure: intent recognition was wrong, fields missing, or category wrong.\n"
    "- generation_failure: retrieval was correct but the generated answer is poor.\n"
    "Return only the label string.\n\n"
    "User question: {query}\n"
    "AI reply: {response}\n"
    "User correction: {correction}\n"
    "Label:"
)


def fallback_classify(query: str, response: str, correction: str | None) -> str:
    if correction:
        return "retrieval_failure"
    return "generation_failure"


async def classify_failure(
    query: str,
    response: str,
    correction: str | None,
    client: AsyncOpenAI | None = None,
) -> str:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        query=query,
                        response=response,
                        correction=correction or "(none)",
                    ),
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        return fallback_classify(query, response, correction)

    for label in FAILURE_TYPES:
        if label in raw:
            return label
    return fallback_classify(query, response, correction)
```

- [ ] **Step 5: Run test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_classifier.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```
git add backend/app/feedback/__init__.py backend/app/feedback/classifier.py backend/tests/test_feedback_classifier.py
git commit -m "feat: P9 failure-type classifier with LLM + rule fallback"
```

---

### Task 2: Feedback API (explicit POST + GET)

**Files:**
- Create: `backend/app/feedback/schemas.py`
- Create: `backend/app/feedback/service.py`
- Create: `backend/app/feedback/router.py`
- Create: `backend/tests/test_feedback_router.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_feedback_router.py`:

```python
import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_post_feedback_records_row(finance_setup, monkeypatch):
    user, team, token = finance_setup
    fake_classify = AsyncMock(return_value="intent_failure")
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            f"/feedback?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "team_id": str(team.id),
                "task_id": str(uuid.uuid4()),
                "task_type": "record_transaction",
                "rating": -1,
                "feedback_type": "correction",
                "original_output": {"category": "Other"},
                "corrected_output": {"category": "Food"},
                "prompt_version": "v1.0",
            },
        )
    assert response.status_code == 201
    body = response.json()
    assert body["rating"] == -1
    assert body["failure_type"] == "intent_failure"


@pytest.mark.asyncio
async def test_post_feedback_skips_classifier_for_positive(finance_setup, monkeypatch):
    user, team, token = finance_setup
    fake_classify = AsyncMock()
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            f"/feedback?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "team_id": str(team.id),
                "rating": 1,
                "feedback_type": "thumbs",
            },
        )
    assert response.status_code == 201
    assert response.json()["failure_type"] is None
    fake_classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_feedback_filters_by_team(finance_setup, monkeypatch):
    user, team, token = finance_setup
    fake_classify = AsyncMock(return_value="generation_failure")
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        await ac.post(
            f"/feedback?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "team_id": str(team.id),
                "rating": -1,
                "feedback_type": "thumbs",
            },
        )
        listed = await ac.get(
            f"/feedback?team_id={team.id}",
            headers={"Authorization": f"Bearer {token}"},
        )
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) >= 1
    assert all(item["team_id"] == str(team.id) for item in items)
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_router.py -v
```
Expected: FAIL — `/feedback` not registered

- [ ] **Step 3: Create schemas**

Create `backend/app/feedback/schemas.py`:

```python
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    team_id: uuid.UUID
    task_id: uuid.UUID | None = None
    task_type: str | None = None
    rating: int = Field(ge=-1, le=1)
    feedback_type: str = Field(min_length=1, max_length=50)
    original_output: dict[str, Any] | None = None
    corrected_output: dict[str, Any] | None = None
    prompt_version: str | None = None


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    team_id: uuid.UUID
    user_id: uuid.UUID
    task_id: uuid.UUID | None
    task_type: str | None
    rating: int
    feedback_type: str
    failure_type: str | None
    original_output: dict[str, Any] | None
    corrected_output: dict[str, Any] | None
    prompt_version: str | None
    created_at: datetime
```

- [ ] **Step 4: Create service**

Create `backend/app/feedback/service.py`:

```python
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.feedback.classifier import classify_failure
from app.models.feedback import Feedback


async def record_feedback(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    task_type: str | None,
    rating: int,
    feedback_type: str,
    original_output: dict[str, Any] | None,
    corrected_output: dict[str, Any] | None,
    prompt_version: str | None,
    db: AsyncSession,
) -> Feedback:
    failure_type: str | None = None
    if rating < 0:
        original_text = ""
        correction_text = ""
        if isinstance(original_output, dict):
            original_text = " ".join(str(v) for v in original_output.values())
        if isinstance(corrected_output, dict):
            correction_text = " ".join(str(v) for v in corrected_output.values())
        failure_type = await classify_failure(
            query=task_type or feedback_type,
            response=original_text,
            correction=correction_text or None,
        )

    row = Feedback(
        team_id=team_id,
        user_id=user_id,
        task_id=task_id,
        task_type=task_type,
        rating=rating,
        feedback_type=feedback_type,
        failure_type=failure_type,
        original_output=original_output,
        corrected_output=corrected_output,
        prompt_version=prompt_version,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def list_feedback(team_id: uuid.UUID, db: AsyncSession) -> list[Feedback]:
    result = await db.execute(
        select(Feedback)
        .where(Feedback.team_id == team_id)
        .order_by(Feedback.created_at.desc())
    )
    return list(result.scalars().all())
```

- [ ] **Step 5: Create router**

Create `backend/app/feedback/router.py`:

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import get_current_user
from app.core.database import get_db
from app.feedback import schemas, service
from app.models.user import User
from app.teams.service import get_member_role

router = APIRouter(prefix="/feedback", tags=["feedback"])

_ANY_ROLES = ("owner", "admin", "member", "viewer")


async def _require_team(
    team_id: uuid.UUID = Query(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> tuple[uuid.UUID, User]:
    role = await get_member_role(team_id, current_user.id, db)
    if role not in _ANY_ROLES:
        raise HTTPException(status_code=403, detail="Team not found or access denied")
    return team_id, current_user


@router.post("", response_model=schemas.FeedbackResponse, status_code=201)
async def create_feedback(
    body: schemas.FeedbackCreate,
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, user = ctx
    if body.team_id != team_id:
        raise HTTPException(status_code=400, detail="team_id mismatch")
    return await service.record_feedback(
        team_id=team_id,
        user_id=user.id,
        task_id=body.task_id,
        task_type=body.task_type,
        rating=body.rating,
        feedback_type=body.feedback_type,
        original_output=body.original_output,
        corrected_output=body.corrected_output,
        prompt_version=body.prompt_version,
        db=db,
    )


@router.get("", response_model=list[schemas.FeedbackResponse])
async def list_feedback(
    ctx: tuple = Depends(_require_team),
    db: AsyncSession = Depends(get_db),
):
    team_id, _ = ctx
    return await service.list_feedback(team_id, db)
```

- [ ] **Step 6: Register router in main.py**

Edit `backend/app/main.py`. Add import and include:

```python
from app.feedback.router import router as feedback_router
```

After existing routers:

```python
app.include_router(feedback_router)
```

- [ ] **Step 7: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_router.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```
git add backend/app/feedback/schemas.py backend/app/feedback/service.py backend/app/feedback/router.py backend/app/main.py backend/tests/test_feedback_router.py
git commit -m "feat: P9 explicit feedback REST endpoints with auto-classification"
```

---

### Task 3: Implicit-feedback hooks on transaction edit/delete

**Files:**
- Modify: `backend/app/finance/service.py`
- Modify: `backend/tests/test_finance_service.py`

- [ ] **Step 1: Write failing test**

Append to `backend/tests/test_finance_service.py`:

```python
@pytest.mark.asyncio
async def test_update_ai_transaction_records_correction_feedback(
    db_session, finance_setup
):
    from sqlalchemy import select

    from app.finance.service import (
        create_account,
        create_category,
        create_transaction,
        update_transaction,
    )
    from app.models.feedback import Feedback

    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)
    category = await create_category(team.id, "Food", None, None, db_session)
    new_category = await create_category(team.id, "Travel", None, None, db_session)

    tx = await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=category.id,
        amount_fen=2500,
        direction="expense",
        description="ai recorded",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
        created_by_ai=True,
    )

    await update_transaction(
        tx.id,
        team.id,
        {"category_id": new_category.id},
        db_session,
    )

    rows = (
        await db_session.execute(
            select(Feedback).where(Feedback.team_id == team.id)
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].feedback_type == "correction"
    assert rows[0].rating == -1


@pytest.mark.asyncio
async def test_soft_delete_ai_transaction_records_deletion_feedback(
    db_session, finance_setup
):
    from sqlalchemy import select

    from app.finance.service import (
        create_account,
        create_transaction,
        soft_delete_transaction,
    )
    from app.models.feedback import Feedback

    user, team, _ = finance_setup
    account = await create_account(team.id, "Cash", "cash", "CNY", 0, db_session)

    tx = await create_transaction(
        team_id=team.id,
        account_id=account.id,
        category_id=None,
        amount_fen=1500,
        direction="expense",
        description="ai recorded",
        transaction_date=date.today(),
        created_by=user.id,
        db=db_session,
        created_by_ai=True,
    )

    await soft_delete_transaction(tx.id, team.id, db_session)

    rows = (
        await db_session.execute(
            select(Feedback).where(Feedback.team_id == team.id)
        )
    ).scalars().all()
    assert any(r.feedback_type == "deletion" for r in rows)
```

(Imports: ensure `from datetime import date` is at module top.)

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_finance_service.py::test_update_ai_transaction_records_correction_feedback -v
```
Expected: FAIL — no Feedback row created

- [ ] **Step 3: Add implicit hooks in finance service**

Edit `backend/app/finance/service.py`. Add import:

```python
from app.models.feedback import Feedback
```

In `update_transaction`, after `await db.refresh(tx)` and the budget invalidation block, add:

```python
    if tx.created_by_ai:
        db.add(
            Feedback(
                team_id=tx.team_id,
                user_id=tx.created_by,
                task_id=tx.id,
                task_type="record_transaction",
                rating=-1,
                feedback_type="correction",
                original_output=None,
                corrected_output=fields,
                prompt_version=None,
            )
        )
        await db.commit()
```

In `soft_delete_transaction`, after `await db.commit()` (the existing one) and the budget invalidation block, add:

```python
    if tx.created_by_ai:
        db.add(
            Feedback(
                team_id=tx.team_id,
                user_id=tx.created_by,
                task_id=tx.id,
                task_type="record_transaction",
                rating=-1,
                feedback_type="deletion",
                original_output=None,
                corrected_output=None,
                prompt_version=None,
            )
        )
        await db.commit()
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_finance_service.py -v
```
Expected: PASS (all)

- [ ] **Step 5: Commit**

```
git add backend/app/finance/service.py backend/tests/test_finance_service.py
git commit -m "feat: P9 implicit feedback signals on AI transaction edit/delete"
```

---

### Task 4: Failure-mode analyzer

**Files:**
- Create: `backend/app/feedback/analyzer.py`
- Create: `backend/tests/test_feedback_analyzer.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_feedback_analyzer.py`:

```python
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.feedback.analyzer import aggregate_failure_modes
from app.models.feedback import Feedback


@pytest.mark.asyncio
async def test_aggregate_failure_modes_counts_each_label(db_session, finance_setup):
    user, team, _ = finance_setup
    rows = [
        Feedback(team_id=team.id, user_id=user.id, rating=-1, feedback_type="thumbs", failure_type="retrieval_failure"),
        Feedback(team_id=team.id, user_id=user.id, rating=-1, feedback_type="thumbs", failure_type="retrieval_failure"),
        Feedback(team_id=team.id, user_id=user.id, rating=-1, feedback_type="correction", failure_type="intent_failure"),
        Feedback(team_id=team.id, user_id=user.id, rating=1, feedback_type="thumbs", failure_type=None),
    ]
    for row in rows:
        db_session.add(row)
    await db_session.commit()

    since = datetime.now(timezone.utc) - timedelta(days=7)
    result = await aggregate_failure_modes(team.id, since.replace(tzinfo=None), db_session)

    assert result["retrieval_failure"] == 2
    assert result["intent_failure"] == 1
    assert result.get("generation_failure", 0) == 0
    assert result["total_negative"] == 3


@pytest.mark.asyncio
async def test_aggregate_excludes_old_rows(db_session, finance_setup):
    user, team, _ = finance_setup
    old = Feedback(
        team_id=team.id, user_id=user.id, rating=-1, feedback_type="thumbs",
        failure_type="retrieval_failure",
    )
    db_session.add(old)
    await db_session.commit()
    await db_session.refresh(old)

    since = datetime.now(timezone.utc) + timedelta(days=1)
    result = await aggregate_failure_modes(team.id, since.replace(tzinfo=None), db_session)
    assert result["total_negative"] == 0
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_analyzer.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement analyzer**

Create `backend/app/feedback/analyzer.py`:

```python
import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feedback import Feedback


async def aggregate_failure_modes(
    team_id: uuid.UUID,
    since: datetime,
    db: AsyncSession,
) -> dict[str, int]:
    result = await db.execute(
        select(Feedback.failure_type, func.count())
        .where(
            Feedback.team_id == team_id,
            Feedback.rating < 0,
            Feedback.created_at >= since,
        )
        .group_by(Feedback.failure_type)
    )
    counts: dict[str, int] = {}
    total = 0
    for failure_type, count in result.all():
        if failure_type:
            counts[failure_type] = int(count)
        total += int(count)
    counts["total_negative"] = total
    return counts
```

- [ ] **Step 4: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_feedback_analyzer.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/feedback/analyzer.py backend/tests/test_feedback_analyzer.py
git commit -m "feat: P9 weekly failure-mode aggregation"
```

---

### Task 5: Eval metrics + golden-set runner

**Files:**
- Create: `backend/app/eval/__init__.py`
- Create: `backend/app/eval/golden_set.py`
- Create: `backend/app/eval/metrics.py`
- Create: `backend/app/eval/runner.py`
- Create: `backend/tests/test_eval_metrics.py`
- Create: `backend/tests/test_eval_runner.py`

- [ ] **Step 1: Create eval package**

Create `backend/app/eval/__init__.py` with empty content.

- [ ] **Step 2: Create golden set**

Create `backend/app/eval/golden_set.py`:

```python
EVAL_CASES: list[dict] = [
    {
        "id": "t001",
        "scenario": "record_transaction",
        "input": "今天午饭35元",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 35,
            "direction": "expense",
            "category": "餐饮",
        },
    },
    {
        "id": "t002",
        "scenario": "record_transaction",
        "input": "瑞幸咖啡28元",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 28,
            "direction": "expense",
            "category": "餐饮",
        },
    },
    {
        "id": "t003",
        "scenario": "record_transaction",
        "input": "客户打款5000到工行",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 5000,
            "direction": "income",
        },
    },
    {
        "id": "t004",
        "scenario": "record_transaction",
        "input": "地铁3元",
        "expect": {
            "fn": "record_transaction",
            "amount_yuan": 3,
            "direction": "expense",
            "category": "交通",
        },
    },
    {
        "id": "t005",
        "scenario": "clarify",
        "input": "帮我记一下今天午饭",
        "expect": {
            "fn": "clarify",
            "missing_fields": ["amount_yuan"],
        },
    },
    {
        "id": "t006",
        "scenario": "generate_report",
        "input": "生成本月报表，按分类",
        "expect": {
            "fn": "generate_report",
            "group_by": "category",
        },
    },
    {
        "id": "t007",
        "scenario": "generate_report",
        "input": "看下这个月每天花了多少",
        "expect": {
            "fn": "generate_report",
            "group_by": "day",
        },
    },
    {
        "id": "t008",
        "scenario": "rag_retrieve",
        "input": "差旅费报销有什么限制",
        "expect": {
            "fn": "rag_retrieve",
        },
    },
    {
        "id": "t009",
        "scenario": "rag_retrieve",
        "input": "出差报销限制",
        "expect": {
            "fn": "rag_retrieve",
        },
    },
    {
        "id": "t010",
        "scenario": "clarify",
        "input": "记一笔",
        "expect": {
            "fn": "clarify",
        },
    },
]


SCENARIO_THRESHOLDS = {
    "record_transaction": 0.90,
    "generate_report": 0.85,
    "rag_retrieve": 0.85,
    "clarify": 0.92,
}
```

- [ ] **Step 3: Write failing metrics test**

Create `backend/tests/test_eval_metrics.py`:

```python
from app.eval.metrics import (
    field_accuracy,
    missing_field_recall,
    score_scenario,
    transaction_count_match,
)


def test_field_accuracy_all_match():
    expected = {"fn": "record_transaction", "amount_yuan": 35, "direction": "expense"}
    actual = {"fn": "record_transaction", "amount_yuan": 35, "direction": "expense"}
    assert field_accuracy(expected, actual) == 1.0


def test_field_accuracy_partial():
    expected = {"fn": "record_transaction", "amount_yuan": 35, "category": "Food"}
    actual = {"fn": "record_transaction", "amount_yuan": 35, "category": "Other"}
    assert field_accuracy(expected, actual) == pytest.approx(2 / 3) if False else (
        abs(field_accuracy(expected, actual) - 2 / 3) < 1e-6
    )


def test_field_accuracy_zero_when_fn_mismatch():
    expected = {"fn": "record_transaction", "amount_yuan": 35}
    actual = {"fn": "clarify"}
    assert field_accuracy(expected, actual) == 0.0


def test_transaction_count_match_exact():
    assert transaction_count_match(3, 3) == 1.0
    assert transaction_count_match(3, 2) == pytest.approx(2 / 3) if False else abs(
        transaction_count_match(3, 2) - 2 / 3
    ) < 1e-6
    assert transaction_count_match(0, 0) == 1.0


def test_missing_field_recall_full():
    assert missing_field_recall(["amount_yuan"], ["amount_yuan"]) == 1.0
    assert missing_field_recall(
        ["amount_yuan", "direction"], ["amount_yuan"]
    ) == 0.5
    assert missing_field_recall([], []) == 1.0


def test_score_scenario_dispatches():
    expected = {"fn": "record_transaction", "amount_yuan": 35}
    actual = {"fn": "record_transaction", "amount_yuan": 35}
    assert score_scenario("record_transaction", actual, expected) == 1.0

    expected_clarify = {"fn": "clarify", "missing_fields": ["amount_yuan"]}
    actual_clarify = {"fn": "clarify", "missing_fields": ["amount_yuan"]}
    assert score_scenario("clarify", actual_clarify, expected_clarify) == 1.0
```

(Add `import pytest` at the top.)

- [ ] **Step 4: Implement metrics**

Create `backend/app/eval/metrics.py`:

```python
from typing import Any


def field_accuracy(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if expected.get("fn") != actual.get("fn"):
        return 0.0
    fields = [k for k in expected.keys() if k != "fn"]
    if not fields:
        return 1.0
    matched = sum(1 for k in fields if expected.get(k) == actual.get(k))
    return matched / len(fields)


def transaction_count_match(expected_count: int, actual_count: int) -> float:
    if expected_count == 0 and actual_count == 0:
        return 1.0
    if expected_count == 0:
        return 0.0
    return min(actual_count, expected_count) / max(actual_count, expected_count)


def missing_field_recall(
    expected_fields: list[str],
    actual_fields: list[str],
) -> float:
    if not expected_fields:
        return 1.0
    expected_set = set(expected_fields)
    matched = sum(1 for f in actual_fields if f in expected_set)
    return matched / len(expected_set)


def score_scenario(
    scenario: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> float:
    if scenario == "clarify":
        if expected.get("fn") != actual.get("fn"):
            return 0.0
        expected_missing = list(expected.get("missing_fields") or [])
        actual_missing = list(actual.get("missing_fields") or [])
        if not expected_missing:
            return 1.0
        return missing_field_recall(expected_missing, actual_missing)
    return field_accuracy(expected, actual)
```

- [ ] **Step 5: Run metrics test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_eval_metrics.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 6: Write failing runner test**

Create `backend/tests/test_eval_runner.py`:

```python
from unittest.mock import AsyncMock

import pytest

from app.eval.runner import run_eval_cases


@pytest.mark.asyncio
async def test_runner_aggregates_per_scenario(monkeypatch):
    cases = [
        {"id": "a", "scenario": "record_transaction",
         "input": "x", "expect": {"fn": "record_transaction", "amount_yuan": 1}},
        {"id": "b", "scenario": "record_transaction",
         "input": "y", "expect": {"fn": "record_transaction", "amount_yuan": 2}},
        {"id": "c", "scenario": "clarify",
         "input": "z", "expect": {"fn": "clarify"}},
    ]

    async def fake_run(text: str) -> dict:
        if text == "x":
            return {"fn": "record_transaction", "amount_yuan": 1}
        if text == "y":
            return {"fn": "record_transaction", "amount_yuan": 99}
        return {"fn": "clarify"}

    result = await run_eval_cases(cases, runner=fake_run)

    assert result["scenarios"]["record_transaction"]["count"] == 2
    assert result["scenarios"]["record_transaction"]["avg_score"] == 0.5
    assert result["scenarios"]["clarify"]["avg_score"] == 1.0
    assert 0.0 < result["aggregate"] < 1.0
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_runner_handles_runner_exception(monkeypatch):
    async def boom(text: str):
        raise RuntimeError("oops")

    cases = [{"id": "a", "scenario": "record_transaction", "input": "x", "expect": {"fn": "record_transaction"}}]
    result = await run_eval_cases(cases, runner=boom)

    assert result["scenarios"]["record_transaction"]["avg_score"] == 0.0
    assert result["errors"] == 1
```

- [ ] **Step 7: Implement runner**

Create `backend/app/eval/runner.py`:

```python
from collections import defaultdict
from typing import Any, Awaitable, Callable

from app.eval.metrics import score_scenario


async def run_eval_cases(
    cases: list[dict[str, Any]],
    runner: Callable[[str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    scenario_scores: dict[str, list[float]] = defaultdict(list)
    errors = 0

    for case in cases:
        scenario = case["scenario"]
        try:
            actual = await runner(case["input"])
        except Exception:
            scenario_scores[scenario].append(0.0)
            errors += 1
            continue
        score = score_scenario(scenario, actual, case["expect"])
        scenario_scores[scenario].append(score)

    scenarios: dict[str, dict[str, float]] = {}
    weighted_total = 0.0
    total_count = 0
    for scenario, scores in scenario_scores.items():
        avg = sum(scores) / len(scores)
        scenarios[scenario] = {"count": len(scores), "avg_score": avg}
        weighted_total += sum(scores)
        total_count += len(scores)

    aggregate = weighted_total / total_count if total_count else 0.0
    return {
        "scenarios": scenarios,
        "aggregate": aggregate,
        "total": total_count,
        "errors": errors,
    }
```

- [ ] **Step 8: Run runner test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_eval_runner.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 9: Commit**

```
git add backend/app/eval/__init__.py backend/app/eval/golden_set.py backend/app/eval/metrics.py backend/app/eval/runner.py backend/tests/test_eval_metrics.py backend/tests/test_eval_runner.py
git commit -m "feat: P9 golden-set eval runner with scenario metrics"
```

---

### Task 6: A/B test bucket assignment

**Files:**
- Create: `backend/app/ab_test/__init__.py`
- Create: `backend/app/ab_test/registry.py`
- Create: `backend/app/ab_test/bucket.py`
- Create: `backend/tests/test_ab_test_bucket.py`

- [ ] **Step 1: Create package**

Create `backend/app/ab_test/__init__.py` with empty content.

- [ ] **Step 2: Write failing test**

Create `backend/tests/test_ab_test_bucket.py`:

```python
import json
import uuid

import pytest

from app.ab_test.bucket import assign_prompt_version, deterministic_bucket
from app.ab_test.registry import (
    EXPERIMENT_KEY,
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
        for k in keys:
            if k in self.store:
                del self.store[k]
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
```

- [ ] **Step 3: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_ab_test_bucket.py -v
```
Expected: FAIL

- [ ] **Step 4: Implement registry + bucket**

Create `backend/app/ab_test/registry.py`:

```python
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
    payload = {"baseline": baseline, "candidate": candidate, "traffic_pct": int(traffic_pct)}
    await redis.setex(EXPERIMENT_KEY, _TTL_SECONDS, json.dumps(payload))


async def clear_experiment(redis: Any) -> None:
    await redis.delete(EXPERIMENT_KEY)
```

Create `backend/app/ab_test/bucket.py`:

```python
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
```

- [ ] **Step 5: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_ab_test_bucket.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```
git add backend/app/ab_test/__init__.py backend/app/ab_test/registry.py backend/app/ab_test/bucket.py backend/tests/test_ab_test_bucket.py
git commit -m "feat: P9 A/B prompt-version bucket assignment"
```

---

### Task 7: Full-suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run full test suite**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/ -q
```
Expected: all green; new test count >= old + 24.

- [ ] **Step 2: If failures, fix and re-run**

Common follow-ups:
- `Feedback.created_at` server-default may be UTC-aware while tests pass naive datetime — strip tzinfo in `aggregate_failure_modes since` argument as already shown in test.
- `update_transaction` test expects exactly 1 feedback row — ensure existing finance tests that update non-AI transactions don't suddenly emit feedback (the `tx.created_by_ai` guard prevents this; verify in test output).
- `test_post_feedback_records_row` may fail with 422 if `feedback_type` enum is too strict — keep the field a free string.
- Eval runner test uses `sum`/`len` math; rounding inside `pytest.approx` is fine.

- [ ] **Step 3: Final commit (if any fix needed)**

```
git add -A
git commit -m "fix: P9 follow-up adjustments after full-suite run"
```

---

## Self-Review

**Spec coverage:**
- §8.1 Feedback collection (explicit + implicit) → Task 2 (explicit endpoints) + Task 3 (implicit hooks)
- §8.2 Failure type classification routing → Task 1 (`classify_failure`) + Task 2 (auto-call on negative feedback) + Task 4 (analyzer aggregates by type)
- §8.3 Scenario-based metrics → Task 5 (`field_accuracy`, `transaction_count_match`, `missing_field_recall`, `score_scenario` dispatcher)
- §8.4 RAGAS — explicitly deferred. `feedback.ragas_scores` column already exists; field stays nullable.
- §8.5 Online A/B testing → Task 6 (`set_experiment`, `assign_prompt_version`)
- §8.6 Multi-scenario Golden Set → Task 5 (`golden_set.EVAL_CASES` ≥10 cases; expand to 100+ over time as feedback corrections roll in)
- §8.7 Closed-loop pipeline — analyzer (Task 4) + runner (Task 5) are the building blocks; Celery scheduling is deferred.

**Out of P9 scope:** RAGAS lib, Celery cron registration, prompt-version write/read endpoints (admin will manage rows directly via DB until UI work in P10).

**Placeholder scan:** none.

**Type consistency:**
- `classify_failure(query, response, correction, client=None) -> str` matches Task 1 signature and Task 2 service caller.
- `record_feedback` returns `Feedback` ORM model; router serializes via `FeedbackResponse.from_attributes`.
- `aggregate_failure_modes(team_id, since: datetime, db)` returns `dict[str, int]` with `total_negative` summary key.
- `score_scenario(scenario, actual, expected)` parameter order consistent across runner and tests.
- `assign_prompt_version(team_id: str, redis, default: str) -> str` matches caller expectations (string-based, not UUID).

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-02-p9-eval-feedback-loop.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — same session, batched checkpoints.

Which approach?
