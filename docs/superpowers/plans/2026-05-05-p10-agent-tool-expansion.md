# P10 Agent Tool Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Agent tool catalog from 4 → 8 tools (spec §4.2): add `record_batch`, `query_transactions`, `analyze_spending`, `check_budget`. Wire each into the OpenAI Function-Calling layer, executor dispatch, prompt few-shots, response builder, and golden set so a single user utterance like "今天早饭12午饭35打车45" produces 3 transactions in one round-trip (target: -60% manual logging cost).

**Architecture:** Pure backend extension building on the existing `agent/{tools,llm,executor}.py` skeleton.
- `agent/tools.py` grows new tool schemas + extended `AgentIntent` literal + rule-fallback hooks for new intents.
- `agent/llm.py` `normalize_intent` dispatcher gains 4 new normalizers; each returns a fully-typed `AgentIntent` payload.
- `agent/executor.py` dispatcher gains 4 new branches: `_record_batch` reuses `finance_service.create_transaction` per item under a single DB transaction; `_query_transactions` thin wrapper over `finance_service.list_transactions`; `_analyze_spending` runs two grouped SQL aggregates (current period + comparison period) and returns deltas; `_check_budget` reuses `finance_service.get_budget_usage`.
- `chat/service.py` `_build_agent_response` gains 4 new branches with concise Chinese summaries.
- `agent/prompt.py` `FEW_SHOT_EXAMPLES` extended with 4 new pairs (batch, query, analyze, budget) keeping the prefix byte-stable for OpenAI prompt cache.
- `eval/golden_set.py` extends `EVAL_CASES` with 8 new cases covering new scenarios.

**Tech Stack:** FastAPI + SQLAlchemy 2.0 (existing), OpenAI Function Calling (`tool_choice="required"`), pgvector ignored (text-only intent parsing). No new dependencies, no new tables.

**Out of P10 scope** (defer):
- LangGraph state-machine refactor — current sequential `chat/service.py` flow is sufficient for 8 tools; LangGraph migration is P11 alongside frontend.
- `record_batch` post-record fan-out (`update_category_embedding`, `collect_implicit_feedback` — already partially handled by P9 `created_by_ai` hooks; per-row alert checks reuse existing `_check_budget_alerts`).
- RAGAS scoring on `analyze_spending` — slot already nullable in `feedback.ragas_scores`; wiring waits for P12.
- Frontend integration — pure backend work.

---

## File Structure

**Modified files (under `backend/app/`):**
- `agent/tools.py` — add 4 tool dicts, extend `AgentIntent` literal, add `_BATCH_RE` rule fallback for multi-amount messages
- `agent/llm.py` — extend `normalize_intent` with 4 new branches; add `_normalize_record_batch`, `_normalize_query_transactions`, `_normalize_analyze_spending`, `_normalize_check_budget`
- `agent/executor.py` — add `_record_batch`, `_query_transactions`, `_analyze_spending`, `_check_budget` and wire each into `execute_intent` dispatch
- `agent/prompt.py` — append 4 few-shot pairs to `FEW_SHOT_EXAMPLES`
- `chat/service.py` — extend `_build_agent_response` with 4 new branches
- `eval/golden_set.py` — append 8 new cases to `EVAL_CASES`

**New tests:**
- `backend/tests/test_agent_record_batch.py`
- `backend/tests/test_agent_query_transactions.py`
- `backend/tests/test_agent_analyze_spending.py`
- `backend/tests/test_agent_check_budget.py`

**Modified tests:**
- `backend/tests/test_agent_llm.py` — add normalize cases for 4 new tools (if file exists; create if missing)
- `backend/tests/test_eval_metrics.py` — `score_scenario` already supports new scenarios via `field_accuracy`; add cases for new scenarios

---

### Task 1: `record_batch` tool (multi-transaction in one call)

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/llm.py`
- Modify: `backend/app/agent/executor.py`
- Create: `backend/tests/test_agent_record_batch.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_record_batch.py`:

```python
import uuid
from datetime import date

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import create_account


def test_normalize_record_batch_keeps_three_items():
    intent: AgentIntent = normalize_intent(
        "record_batch",
        {
            "transactions": [
                {"amount_yuan": 12, "direction": "expense", "category": "餐饮",
                 "account_name": "现金", "transaction_date": "2026-05-05",
                 "description": "早饭"},
                {"amount_yuan": 35, "direction": "expense", "category": "餐饮",
                 "account_name": "现金", "transaction_date": "2026-05-05",
                 "description": "午饭"},
                {"amount_yuan": 45, "direction": "expense", "category": "交通",
                 "account_name": "现金", "transaction_date": "2026-05-05",
                 "description": "打车"},
            ]
        },
        original_message="今天早饭12午饭35打车45",
    )
    assert intent["name"] == "record_batch"
    items = intent["arguments"]["transactions"]
    assert len(items) == 3
    assert items[0]["amount_fen"] == 1200
    assert items[2]["category"] == "交通"


def test_normalize_record_batch_falls_back_to_clarify_when_empty():
    intent = normalize_intent("record_batch", {"transactions": []}, original_message="")
    assert intent["name"] == "clarify"


def test_normalize_record_batch_drops_invalid_items():
    intent = normalize_intent(
        "record_batch",
        {
            "transactions": [
                {"amount_yuan": 12, "direction": "expense", "category": "餐饮",
                 "account_name": "现金", "transaction_date": "2026-05-05",
                 "description": "早饭"},
                {"amount_yuan": None, "direction": "expense"},
            ]
        },
        original_message="今天早饭12",
    )
    assert intent["name"] == "record_batch"
    assert len(intent["arguments"]["transactions"]) == 1


@pytest.mark.asyncio
async def test_execute_record_batch_creates_three_transactions(
    db_session, finance_setup
):
    user, team, _ = finance_setup
    await create_account(team.id, "现金", "cash", "CNY", 0, db_session)

    intent: AgentIntent = {
        "name": "record_batch",
        "arguments": {
            "transactions": [
                {"amount_yuan": 12.0, "amount_fen": 1200, "direction": "expense",
                 "category": "餐饮", "account_name": "现金",
                 "transaction_date": "2026-05-05", "description": "早饭"},
                {"amount_yuan": 35.0, "amount_fen": 3500, "direction": "expense",
                 "category": "餐饮", "account_name": "现金",
                 "transaction_date": "2026-05-05", "description": "午饭"},
                {"amount_yuan": 45.0, "amount_fen": 4500, "direction": "expense",
                 "category": "交通", "account_name": "现金",
                 "transaction_date": "2026-05-05", "description": "打车"},
            ]
        },
    }

    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "recorded_batch"
    assert result["count"] == 3
    assert len(result["transaction_ids"]) == 3


@pytest.mark.asyncio
async def test_execute_record_batch_no_account_returns_clarification(
    db_session, finance_setup
):
    user, team, _ = finance_setup
    intent: AgentIntent = {
        "name": "record_batch",
        "arguments": {
            "transactions": [
                {"amount_yuan": 12.0, "amount_fen": 1200, "direction": "expense",
                 "category": "餐饮", "account_name": "现金",
                 "transaction_date": "2026-05-05", "description": "早饭"},
            ]
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "needs_clarification"
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_record_batch.py -v
```
Expected: FAIL — `normalize_intent` does not handle `record_batch`, `execute_intent` has no branch.

- [ ] **Step 3: Add tool schema and Literal entry**

Edit `backend/app/agent/tools.py`. Replace `AgentIntent` Literal:

```python
class AgentIntent(TypedDict):
    name: Literal[
        "record_transaction",
        "record_batch",
        "query_transactions",
        "analyze_spending",
        "check_budget",
        "generate_report",
        "rag_retrieve",
        "clarify",
    ]
    arguments: dict[str, Any]
```

Add tool schema after `RECORD_TRANSACTION_TOOL`:

```python
RECORD_BATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "record_batch",
        "description": (
            "Create multiple finance transactions in a single call when the user "
            "describes several money movements in one message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "amount_yuan": {"type": "number"},
                            "direction": {"type": "string", "enum": ["income", "expense"]},
                            "category": {"type": "string"},
                            "account_name": {"type": "string"},
                            "transaction_date": {"type": "string", "format": "date"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "amount_yuan",
                            "direction",
                            "category",
                            "account_name",
                            "transaction_date",
                            "description",
                        ],
                    },
                }
            },
            "required": ["transactions"],
        },
    },
}
```

(Leave `FINBOT_TOOLS` list alone in this task — Task 5 wires the full list.)

- [ ] **Step 4: Add normalizer in `llm.py`**

Edit `backend/app/agent/llm.py`. In `normalize_intent`, add a branch before the existing `record_transaction` branch:

```python
    if name == "record_batch":
        return _normalize_record_batch(arguments, original_message)
```

Append below `_normalize_record_transaction`:

```python
def _normalize_record_batch(
    arguments: dict[str, Any], original_message: str
) -> AgentIntent:
    raw_items = arguments.get("transactions") or []
    if not isinstance(raw_items, list):
        raw_items = []
    items: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        amount_yuan = _decimal_from_value(raw.get("amount_yuan"))
        if amount_yuan is None:
            continue
        direction = raw.get("direction")
        if direction not in {"income", "expense"}:
            continue
        items.append(
            {
                "amount_yuan": float(amount_yuan),
                "amount_fen": yuan_to_fen(amount_yuan),
                "direction": direction,
                "category": str(raw.get("category") or "Uncategorized"),
                "account_name": str(raw.get("account_name") or "Default"),
                "transaction_date": str(
                    raw.get("transaction_date") or date.today().isoformat()
                ),
                "description": str(raw.get("description") or original_message),
            }
        )
    if not items:
        return {
            "name": "clarify",
            "arguments": {
                "question": "Please describe at least one transaction with amount and direction.",
                "missing_fields": ["transactions"],
            },
        }
    return {"name": "record_batch", "arguments": {"transactions": items}}
```

(`yuan_to_fen` and `_decimal_from_value` already exist in this file.)

- [ ] **Step 5: Add executor branch**

Edit `backend/app/agent/executor.py`. In `execute_intent`, before the existing `generate_report` branch:

```python
    if intent["name"] == "record_batch":
        return await _record_batch(intent["arguments"], team_id, user_id, db)
```

Append below `_record_transaction`:

```python
async def _record_batch(
    args: dict[str, Any],
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    items = args.get("transactions") or []
    if not items:
        return {
            "status": "needs_clarification",
            "message": "No transactions provided.",
            "missing_fields": ["transactions"],
        }

    transaction_ids: list[str] = []
    alert_ids: list[str] = []
    for item in items:
        account = await _resolve_account(team_id, item["account_name"], db)
        if account is None:
            return {
                "status": "needs_clarification",
                "message": "No active account exists for this team yet.",
                "missing_fields": ["account_id"],
            }
        category = await _resolve_category(team_id, item["category"], db)
        transaction = await finance_service.create_transaction(
            team_id=team_id,
            account_id=account.id,
            category_id=category.id if category else None,
            amount_fen=int(item["amount_fen"]),
            direction=item["direction"],
            description=item["description"],
            transaction_date=date.fromisoformat(item["transaction_date"]),
            created_by=user_id,
            db=db,
            created_by_ai=True,
        )
        transaction_ids.append(str(transaction.id))
        for alert_id in await _check_budget_alerts(transaction, team_id, db):
            alert_ids.append(str(alert_id))

    return {
        "status": "recorded_batch",
        "message": f"Recorded {len(transaction_ids)} transactions.",
        "count": len(transaction_ids),
        "transaction_ids": transaction_ids,
        "alert_ids": alert_ids,
    }
```

- [ ] **Step 6: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_record_batch.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/llm.py backend/app/agent/executor.py backend/tests/test_agent_record_batch.py
git commit -m "feat: P10 record_batch tool for multi-transaction single-call"
```

---

### Task 2: `query_transactions` tool

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/llm.py`
- Modify: `backend/app/agent/executor.py`
- Create: `backend/tests/test_agent_query_transactions.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_query_transactions.py`:

```python
from datetime import date, timedelta

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import (
    create_account,
    create_category,
    create_transaction,
)


def test_normalize_query_transactions_defaults_to_current_month():
    intent: AgentIntent = normalize_intent(
        "query_transactions",
        {},
        original_message="查一下这个月的支出",
    )
    assert intent["name"] == "query_transactions"
    args = intent["arguments"]
    today = date.today()
    assert args["date_from"] == today.replace(day=1).isoformat()
    assert args["date_to"] == today.isoformat()
    assert args.get("direction") in {None, "expense", "income"}


def test_normalize_query_transactions_passes_filters():
    intent = normalize_intent(
        "query_transactions",
        {
            "date_from": "2026-04-01",
            "date_to": "2026-04-30",
            "direction": "expense",
            "category": "餐饮",
        },
        original_message="四月餐饮花了多少",
    )
    args = intent["arguments"]
    assert args["date_from"] == "2026-04-01"
    assert args["direction"] == "expense"
    assert args["category"] == "餐饮"


@pytest.mark.asyncio
async def test_execute_query_transactions_returns_filtered_rows(
    db_session, finance_setup
):
    user, team, _ = finance_setup
    account = await create_account(team.id, "现金", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "餐饮", None, None, db_session)
    travel = await create_category(team.id, "交通", None, None, db_session)
    today = date.today()
    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=food.id,
        amount_fen=3500, direction="expense", description="lunch",
        transaction_date=today, created_by=user.id, db=db_session,
    )
    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=travel.id,
        amount_fen=4500, direction="expense", description="taxi",
        transaction_date=today, created_by=user.id, db=db_session,
    )

    intent: AgentIntent = {
        "name": "query_transactions",
        "arguments": {
            "date_from": today.isoformat(),
            "date_to": today.isoformat(),
            "category": "餐饮",
            "direction": "expense",
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "queried"
    assert result["count"] == 1
    assert result["total_fen"] == 3500
    assert len(result["items"]) == 1
    assert result["items"][0]["description"] == "lunch"
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_query_transactions.py -v
```
Expected: FAIL

- [ ] **Step 3: Add tool schema**

Edit `backend/app/agent/tools.py`. Append after `RECORD_BATCH_TOOL`:

```python
QUERY_TRANSACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "query_transactions",
        "description": "List historical transactions filtered by period, category, account, or direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "format": "date"},
                "date_to": {"type": "string", "format": "date"},
                "category": {"type": "string"},
                "account_name": {"type": "string"},
                "direction": {"type": "string", "enum": ["income", "expense"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": [],
        },
    },
}
```

- [ ] **Step 4: Add normalizer**

Edit `backend/app/agent/llm.py`. In `normalize_intent`, add branch:

```python
    if name == "query_transactions":
        return _normalize_query_transactions(arguments)
```

Append:

```python
def _normalize_query_transactions(arguments: dict[str, Any]) -> AgentIntent:
    today = date.today()
    direction = arguments.get("direction")
    if direction not in {"income", "expense"}:
        direction = None
    raw_limit = arguments.get("limit")
    try:
        limit = int(raw_limit) if raw_limit is not None else 20
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))
    return {
        "name": "query_transactions",
        "arguments": {
            "date_from": str(arguments.get("date_from") or today.replace(day=1).isoformat()),
            "date_to": str(arguments.get("date_to") or today.isoformat()),
            "direction": direction,
            "category": arguments.get("category") or None,
            "account_name": arguments.get("account_name") or None,
            "limit": limit,
        },
    }
```

- [ ] **Step 5: Add executor branch**

Edit `backend/app/agent/executor.py`. In `execute_intent`, add before the `generate_report` branch:

```python
    if intent["name"] == "query_transactions":
        return await _query_transactions(intent["arguments"], team_id, db)
```

Append:

```python
async def _query_transactions(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    date_from = date.fromisoformat(args["date_from"])
    date_to = date.fromisoformat(args["date_to"])
    category_id: uuid.UUID | None = None
    if args.get("category"):
        category = await _resolve_category(team_id, args["category"], db)
        if category is not None:
            category_id = category.id
    account_id: uuid.UUID | None = None
    if args.get("account_name"):
        account = await _resolve_account(team_id, args["account_name"], db)
        if account is not None:
            account_id = account.id

    rows = await finance_service.list_transactions(
        team_id=team_id,
        date_from=date_from,
        date_to=date_to,
        db=db,
        category_id=category_id,
        account_id=account_id,
    )
    direction_filter = args.get("direction")
    if direction_filter in {"income", "expense"}:
        rows = [row for row in rows if row.direction == direction_filter]
    limit = int(args.get("limit") or 20)
    rows = rows[:limit]
    items = [
        {
            "id": str(row.id),
            "amount_fen": row.amount_fen,
            "direction": row.direction,
            "description": row.description,
            "transaction_date": row.transaction_date.isoformat(),
            "category_id": str(row.category_id) if row.category_id else None,
            "account_id": str(row.account_id),
        }
        for row in rows
    ]
    total_fen = sum(item["amount_fen"] for item in items)
    return {
        "status": "queried",
        "message": f"Found {len(items)} transactions, total {total_fen} fen.",
        "count": len(items),
        "total_fen": total_fen,
        "items": items,
    }
```

- [ ] **Step 6: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_query_transactions.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 7: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/llm.py backend/app/agent/executor.py backend/tests/test_agent_query_transactions.py
git commit -m "feat: P10 query_transactions tool with period and filter normalization"
```

---

### Task 3: `analyze_spending` tool (period comparison)

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/llm.py`
- Modify: `backend/app/agent/executor.py`
- Create: `backend/tests/test_agent_analyze_spending.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_analyze_spending.py`:

```python
from datetime import date, timedelta

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import (
    create_account,
    create_category,
    create_transaction,
)


def test_normalize_analyze_spending_defaults():
    intent: AgentIntent = normalize_intent(
        "analyze_spending",
        {},
        original_message="最近花了多少",
    )
    args = intent["arguments"]
    assert intent["name"] == "analyze_spending"
    assert args["period"] == "last_30_days"
    assert args["group_by"] == "category"
    assert args["compare_with"] in {"prev_period", None}


def test_normalize_analyze_spending_custom_dates():
    intent = normalize_intent(
        "analyze_spending",
        {
            "period": "custom",
            "period_start": "2026-04-01",
            "period_end": "2026-04-30",
            "group_by": "day",
            "compare_with": "prev_period",
        },
        original_message="四月每天花多少",
    )
    args = intent["arguments"]
    assert args["period"] == "custom"
    assert args["period_start"] == "2026-04-01"
    assert args["compare_with"] == "prev_period"


@pytest.mark.asyncio
async def test_execute_analyze_spending_compares_two_periods(
    db_session, finance_setup
):
    user, team, _ = finance_setup
    account = await create_account(team.id, "现金", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "餐饮", None, None, db_session)

    today = date.today()
    cur_start = today - timedelta(days=10)
    prev_start = today - timedelta(days=20)
    prev_end = today - timedelta(days=11)

    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=food.id,
        amount_fen=10000, direction="expense", description="cur",
        transaction_date=cur_start, created_by=user.id, db=db_session,
    )
    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=food.id,
        amount_fen=4000, direction="expense", description="prev",
        transaction_date=prev_start, created_by=user.id, db=db_session,
    )

    intent: AgentIntent = {
        "name": "analyze_spending",
        "arguments": {
            "period": "custom",
            "period_start": cur_start.isoformat(),
            "period_end": today.isoformat(),
            "group_by": "category",
            "compare_with": "prev_period",
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "analyzed"
    assert result["current_total_fen"] == 10000
    assert result["compare_total_fen"] == 4000
    assert result["delta_fen"] == 6000
    assert any(r["amount_fen"] == 10000 for r in result["current_rows"])


@pytest.mark.asyncio
async def test_execute_analyze_spending_without_compare(
    db_session, finance_setup
):
    user, team, _ = finance_setup
    account = await create_account(team.id, "现金", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "餐饮", None, None, db_session)
    today = date.today()
    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=food.id,
        amount_fen=5000, direction="expense", description="x",
        transaction_date=today, created_by=user.id, db=db_session,
    )

    intent: AgentIntent = {
        "name": "analyze_spending",
        "arguments": {
            "period": "custom",
            "period_start": today.isoformat(),
            "period_end": today.isoformat(),
            "group_by": "category",
            "compare_with": None,
        },
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "analyzed"
    assert result["compare_total_fen"] is None
    assert result["delta_fen"] is None
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_analyze_spending.py -v
```
Expected: FAIL

- [ ] **Step 3: Add tool schema**

Edit `backend/app/agent/tools.py`. Append after `QUERY_TRANSACTIONS_TOOL`:

```python
ANALYZE_SPENDING_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_spending",
        "description": (
            "Analyze spending trends across a period and optionally compare to "
            "the previous equal-length period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["last_7_days", "last_30_days", "this_month", "custom"],
                },
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "group_by": {
                    "type": "string",
                    "enum": ["category", "account", "day", "week"],
                },
                "compare_with": {
                    "type": "string",
                    "enum": ["prev_period"],
                },
            },
            "required": ["period"],
        },
    },
}
```

- [ ] **Step 4: Add normalizer**

Edit `backend/app/agent/llm.py`. In `normalize_intent`, add branch:

```python
    if name == "analyze_spending":
        return _normalize_analyze_spending(arguments)
```

Append:

```python
def _normalize_analyze_spending(arguments: dict[str, Any]) -> AgentIntent:
    today = date.today()
    period = arguments.get("period") or "last_30_days"
    if period not in {"last_7_days", "last_30_days", "this_month", "custom"}:
        period = "last_30_days"
    if period == "custom":
        period_start = str(
            arguments.get("period_start") or today.replace(day=1).isoformat()
        )
        period_end = str(arguments.get("period_end") or today.isoformat())
    elif period == "last_7_days":
        period_start = (today - __import__("datetime").timedelta(days=6)).isoformat()
        period_end = today.isoformat()
    elif period == "this_month":
        period_start = today.replace(day=1).isoformat()
        period_end = today.isoformat()
    else:
        period_start = (today - __import__("datetime").timedelta(days=29)).isoformat()
        period_end = today.isoformat()

    group_by = arguments.get("group_by") or "category"
    if group_by not in {"category", "account", "day", "week"}:
        group_by = "category"

    compare_with = arguments.get("compare_with")
    if compare_with not in {"prev_period"}:
        compare_with = None

    return {
        "name": "analyze_spending",
        "arguments": {
            "period": period,
            "period_start": period_start,
            "period_end": period_end,
            "group_by": group_by,
            "compare_with": compare_with,
        },
    }
```

(Add `from datetime import timedelta` at the top of `llm.py` if not already imported, then replace `__import__("datetime").timedelta(...)` with `timedelta(...)`.)

- [ ] **Step 5: Add executor branch**

Edit `backend/app/agent/executor.py`. Add `from datetime import date, timedelta` at top if `timedelta` is not yet imported. In `execute_intent`, add branch:

```python
    if intent["name"] == "analyze_spending":
        return await _analyze_spending(intent["arguments"], team_id, db)
```

Append:

```python
async def _analyze_spending(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    period_start = date.fromisoformat(args["period_start"])
    period_end = date.fromisoformat(args["period_end"])
    group_by = args["group_by"]
    current_rows = await _aggregate_expense(team_id, period_start, period_end, group_by, db)
    current_total = sum(row["amount_fen"] for row in current_rows)

    compare_total: int | None = None
    compare_rows: list[dict[str, Any]] | None = None
    delta: int | None = None
    if args.get("compare_with") == "prev_period":
        span_days = (period_end - period_start).days + 1
        prev_end = period_start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=span_days - 1)
        compare_rows = await _aggregate_expense(
            team_id, prev_start, prev_end, group_by, db
        )
        compare_total = sum(row["amount_fen"] for row in compare_rows)
        delta = current_total - compare_total

    return {
        "status": "analyzed",
        "message": (
            f"Spending {current_total} fen from {period_start} to {period_end}."
        ),
        "group_by": group_by,
        "current_total_fen": current_total,
        "current_rows": current_rows,
        "compare_total_fen": compare_total,
        "compare_rows": compare_rows,
        "delta_fen": delta,
    }


async def _aggregate_expense(
    team_id: uuid.UUID,
    period_start: date,
    period_end: date,
    group_by: str,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    if group_by == "category":
        group_column = Transaction.category_id
    elif group_by == "account":
        group_column = Transaction.account_id
    elif group_by == "week":
        group_column = func.date_trunc("week", Transaction.transaction_date)
    else:
        group_column = Transaction.transaction_date
    result = await db.execute(
        select(
            group_column.label("group_key"),
            func.coalesce(func.sum(Transaction.amount_fen), 0).label("amount_fen"),
        )
        .where(
            and_(
                Transaction.team_id == team_id,
                Transaction.direction == "expense",
                Transaction.transaction_date >= period_start,
                Transaction.transaction_date <= period_end,
                Transaction.deleted_at.is_(None),
            )
        )
        .group_by(group_column)
        .order_by(group_column)
    )
    rows: list[dict[str, Any]] = []
    for group_key, amount_fen in result.all():
        if isinstance(group_key, date):
            key = group_key.isoformat()
        elif group_key is None:
            key = None
        else:
            key = str(group_key)
        rows.append({"group_key": key, "amount_fen": int(amount_fen)})
    return rows
```

- [ ] **Step 6: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_analyze_spending.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/llm.py backend/app/agent/executor.py backend/tests/test_agent_analyze_spending.py
git commit -m "feat: P10 analyze_spending tool with period comparison aggregation"
```

---

### Task 4: `check_budget` tool

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/llm.py`
- Modify: `backend/app/agent/executor.py`
- Create: `backend/tests/test_agent_check_budget.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_check_budget.py`:

```python
from datetime import date

import pytest

from app.agent.executor import execute_intent
from app.agent.llm import normalize_intent
from app.agent.tools import AgentIntent
from app.finance.service import (
    create_account,
    create_budget,
    create_category,
    create_transaction,
)


def test_normalize_check_budget_returns_intent():
    intent: AgentIntent = normalize_intent(
        "check_budget",
        {"category": "餐饮"},
        original_message="餐饮预算还剩多少",
    )
    assert intent["name"] == "check_budget"
    assert intent["arguments"]["category"] == "餐饮"


def test_normalize_check_budget_without_category():
    intent = normalize_intent("check_budget", {}, original_message="预算情况")
    assert intent["name"] == "check_budget"
    assert intent["arguments"]["category"] is None


@pytest.mark.asyncio
async def test_execute_check_budget_returns_usage(db_session, finance_setup):
    user, team, _ = finance_setup
    account = await create_account(team.id, "现金", "cash", "CNY", 0, db_session)
    food = await create_category(team.id, "餐饮", None, None, db_session)
    budget = await create_budget(
        team.id, food.id, 100000, "monthly", 0.8, db_session
    )
    await create_transaction(
        team_id=team.id, account_id=account.id, category_id=food.id,
        amount_fen=20000, direction="expense", description="lunch",
        transaction_date=date.today(), created_by=user.id, db=db_session,
    )

    intent: AgentIntent = {
        "name": "check_budget",
        "arguments": {"category": "餐饮"},
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "checked"
    assert len(result["budgets"]) == 1
    item = result["budgets"][0]
    assert item["amount_fen"] == 100000
    assert item["spent_fen"] == 20000
    assert 0 < item["usage_ratio"] < 1


@pytest.mark.asyncio
async def test_execute_check_budget_no_match_returns_empty(db_session, finance_setup):
    user, team, _ = finance_setup
    intent: AgentIntent = {
        "name": "check_budget",
        "arguments": {"category": "餐饮"},
    }
    result = await execute_intent(intent, team.id, user.id, db_session)
    assert result["status"] == "checked"
    assert result["budgets"] == []
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_check_budget.py -v
```
Expected: FAIL

- [ ] **Step 3: Add tool schema**

Edit `backend/app/agent/tools.py`. Append after `ANALYZE_SPENDING_TOOL`:

```python
CHECK_BUDGET_TOOL = {
    "type": "function",
    "function": {
        "name": "check_budget",
        "description": "Report current budget usage, optionally filtered by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
            "required": [],
        },
    },
}
```

- [ ] **Step 4: Add normalizer**

Edit `backend/app/agent/llm.py`. In `normalize_intent`, add branch:

```python
    if name == "check_budget":
        return _normalize_check_budget(arguments)
```

Append:

```python
def _normalize_check_budget(arguments: dict[str, Any]) -> AgentIntent:
    raw_category = arguments.get("category")
    category = str(raw_category).strip() if raw_category else None
    return {
        "name": "check_budget",
        "arguments": {"category": category or None},
    }
```

- [ ] **Step 5: Add executor branch**

Edit `backend/app/agent/executor.py`. In `execute_intent`, add branch:

```python
    if intent["name"] == "check_budget":
        return await _check_budget(intent["arguments"], team_id, db)
```

Append:

```python
async def _check_budget(
    args: dict[str, Any],
    team_id: uuid.UUID,
    db: AsyncSession,
) -> dict[str, Any]:
    category_id: uuid.UUID | None = None
    if args.get("category"):
        category = await _resolve_category(team_id, args["category"], db)
        if category is not None:
            category_id = category.id

    stmt = select(Budget).where(Budget.team_id == team_id, Budget.is_active == True)
    if category_id is not None:
        stmt = stmt.where(Budget.category_id == category_id)
    result = await db.execute(stmt)
    budgets = list(result.scalars().all())

    items: list[dict[str, Any]] = []
    for budget in budgets:
        usage = await finance_service.get_budget_usage(budget.id, team_id, db)
        items.append(
            {
                "budget_id": str(budget.id),
                "category_id": str(budget.category_id) if budget.category_id else None,
                "amount_fen": int(usage["amount_fen"]),
                "spent_fen": int(usage["spent_fen"]),
                "usage_ratio": float(usage["usage_ratio"]),
                "period": usage["period"],
            }
        )
    return {
        "status": "checked",
        "message": f"Found {len(items)} active budget(s).",
        "budgets": items,
    }
```

- [ ] **Step 6: Run tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_check_budget.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/llm.py backend/app/agent/executor.py backend/tests/test_agent_check_budget.py
git commit -m "feat: P10 check_budget tool with category filter"
```

---

### Task 5: Wire `FINBOT_TOOLS` list, response builder, few-shot examples

**Files:**
- Modify: `backend/app/agent/tools.py`
- Modify: `backend/app/agent/prompt.py`
- Modify: `backend/app/chat/service.py`

- [ ] **Step 1: Update `FINBOT_TOOLS`**

Edit `backend/app/agent/tools.py`. Replace the `FINBOT_TOOLS = [...]` line with:

```python
FINBOT_TOOLS = [
    RECORD_TRANSACTION_TOOL,
    RECORD_BATCH_TOOL,
    QUERY_TRANSACTIONS_TOOL,
    ANALYZE_SPENDING_TOOL,
    CHECK_BUDGET_TOOL,
    GENERATE_REPORT_TOOL,
    RAG_RETRIEVE_TOOL,
    CLARIFY_TOOL,
]
```

- [ ] **Step 2: Extend response builder**

Edit `backend/app/chat/service.py`. Replace `_build_agent_response` with:

```python
def _build_agent_response(
    intent: AgentIntent, execution: dict[str, Any] | None = None
) -> str:
    if execution:
        return str(execution["message"])
    if intent["name"] == "record_transaction":
        args = intent["arguments"]
        return (
            f"已解析为{args['direction']}记录：{args['category']} "
            f"{args['amount_fen']} 分，账户 {args['account_name']}。"
        )
    if intent["name"] == "record_batch":
        items = intent["arguments"].get("transactions") or []
        return f"已解析 {len(items)} 笔批量记账。"
    if intent["name"] == "query_transactions":
        args = intent["arguments"]
        return f"已解析为查询：{args['date_from']} 至 {args['date_to']}。"
    if intent["name"] == "analyze_spending":
        args = intent["arguments"]
        return (
            f"已解析为支出分析：{args['period_start']} 至 {args['period_end']}，"
            f"按 {args['group_by']} 汇总。"
        )
    if intent["name"] == "check_budget":
        args = intent["arguments"]
        scope = args.get("category") or "全部分类"
        return f"已解析为预算检查：{scope}。"
    if intent["name"] == "generate_report":
        args = intent["arguments"]
        return f"已解析为报表请求：{args['period_start']} 至 {args['period_end']}，按 {args['group_by']} 汇总。"
    return str(intent["arguments"]["question"])
```

- [ ] **Step 3: Append few-shot examples**

Edit `backend/app/agent/prompt.py`. Append four pairs to `FEW_SHOT_EXAMPLES` (the list literal):

```python
    {
        "role": "user",
        "content": "今天早饭12 午饭35 打车45。",
    },
    {
        "role": "assistant",
        "content": (
            "Record batch: three transactions on today, expense, account_name=Cash. "
            "Items: amount_fen=1200 category=Food & Beverage description=早饭; "
            "amount_fen=3500 category=Food & Beverage description=午饭; "
            "amount_fen=4500 category=Transportation description=打车."
        ),
    },
    {
        "role": "user",
        "content": "查一下这个月餐饮花了多少。",
    },
    {
        "role": "assistant",
        "content": (
            "Query transactions: date_from=first day of current month, "
            "date_to=current date, category=Food & Beverage, direction=expense."
        ),
    },
    {
        "role": "user",
        "content": "最近30天和上一周期相比花了多少？",
    },
    {
        "role": "assistant",
        "content": (
            "Analyze spending: period=last_30_days, group_by=category, "
            "compare_with=prev_period."
        ),
    },
    {
        "role": "user",
        "content": "餐饮预算还剩多少？",
    },
    {
        "role": "assistant",
        "content": "Check budget: category=Food & Beverage.",
    },
```

- [ ] **Step 4: Verify prompt prefix still ≥1024 tokens**

```
C:/Users/henry/.conda/envs/finbot/python.exe -c "from app.agent.prompt import prompt_prefix_token_estimate; print(prompt_prefix_token_estimate())"
```
Expected: integer ≥ 1024.

- [ ] **Step 5: Run agent and chat test files together**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_record_batch.py backend/tests/test_agent_query_transactions.py backend/tests/test_agent_analyze_spending.py backend/tests/test_agent_check_budget.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```
git add backend/app/agent/tools.py backend/app/agent/prompt.py backend/app/chat/service.py
git commit -m "feat: P10 wire 4 new tools into FINBOT_TOOLS, prompt few-shots, response builder"
```

---

### Task 6: Extend Golden Set with new scenarios

**Files:**
- Modify: `backend/app/eval/golden_set.py`
- Modify: `backend/tests/test_eval_metrics.py`

- [ ] **Step 1: Append new eval cases**

Edit `backend/app/eval/golden_set.py`. Append inside `EVAL_CASES` (before the closing `]`):

```python
    {
        "id": "t011",
        "scenario": "record_batch",
        "input": "今天早饭12午饭35打车45",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
    {
        "id": "t012",
        "scenario": "record_batch",
        "input": "买菜20 打车15 咖啡30",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
    {
        "id": "t013",
        "scenario": "query_transactions",
        "input": "查一下这个月餐饮支出",
        "expect": {
            "fn": "query_transactions",
            "direction": "expense",
            "category": "餐饮",
        },
    },
    {
        "id": "t014",
        "scenario": "analyze_spending",
        "input": "这个月餐饮比上个月多花了多少",
        "expect": {
            "fn": "analyze_spending",
            "compare_with": "prev_period",
        },
    },
    {
        "id": "t015",
        "scenario": "analyze_spending",
        "input": "最近7天每天花了多少",
        "expect": {
            "fn": "analyze_spending",
            "period": "last_7_days",
            "group_by": "day",
        },
    },
    {
        "id": "t016",
        "scenario": "check_budget",
        "input": "餐饮预算还剩多少",
        "expect": {
            "fn": "check_budget",
            "category": "餐饮",
        },
    },
    {
        "id": "t017",
        "scenario": "check_budget",
        "input": "看一下整体预算情况",
        "expect": {
            "fn": "check_budget",
        },
    },
    {
        "id": "t018",
        "scenario": "record_batch",
        "input": "上午地铁3，午饭28，下午奶茶15",
        "expect": {
            "fn": "record_batch",
            "count": 3,
        },
    },
```

Add the same scenarios to `SCENARIO_THRESHOLDS` (replace the existing dict literal with):

```python
SCENARIO_THRESHOLDS = {
    "record_transaction": 0.90,
    "record_batch": 0.85,
    "query_transactions": 0.85,
    "generate_report": 0.85,
    "analyze_spending": 0.85,
    "check_budget": 0.85,
    "rag_retrieve": 0.85,
    "clarify": 0.92,
}
```

- [ ] **Step 2: Add metric tests for new scenarios**

Edit `backend/tests/test_eval_metrics.py`. Append:

```python
def test_score_scenario_record_batch_count_field():
    expected = {"fn": "record_batch", "count": 3}
    actual = {"fn": "record_batch", "count": 3}
    assert score_scenario("record_batch", actual, expected) == 1.0

    actual_partial = {"fn": "record_batch", "count": 2}
    assert score_scenario("record_batch", actual_partial, expected) == 0.0


def test_score_scenario_analyze_spending_compare_field():
    expected = {"fn": "analyze_spending", "compare_with": "prev_period"}
    actual = {"fn": "analyze_spending", "compare_with": "prev_period"}
    assert score_scenario("analyze_spending", actual, expected) == 1.0

    actual_wrong_fn = {"fn": "generate_report", "compare_with": "prev_period"}
    assert score_scenario("analyze_spending", actual_wrong_fn, expected) == 0.0


def test_score_scenario_check_budget_category_field():
    expected = {"fn": "check_budget", "category": "餐饮"}
    actual_match = {"fn": "check_budget", "category": "餐饮"}
    actual_other = {"fn": "check_budget", "category": "交通"}
    assert score_scenario("check_budget", actual_match, expected) == 1.0
    assert score_scenario("check_budget", actual_other, expected) == 0.0
```

- [ ] **Step 3: Run eval tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_eval_metrics.py backend/tests/test_eval_runner.py -v
```
Expected: all PASS (existing 6 + 3 new metric tests + 2 runner tests).

- [ ] **Step 4: Commit**

```
git add backend/app/eval/golden_set.py backend/tests/test_eval_metrics.py
git commit -m "feat: P10 extend golden set + metrics for new agent tool scenarios"
```

---

### Task 7: Full-suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run full test suite (Docker required)**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/ -q
```
Expected: all green (DB-required tests need Docker running with `postgres:5433` + `redis:6379`). New tests added: 5 (record_batch) + 3 (query_transactions) + 4 (analyze_spending) + 4 (check_budget) + 3 (eval metrics) = 19. Total should be previous 99 + 19 = 118+ passing.

- [ ] **Step 2: If failures, fix and re-run**

Common follow-ups:
- `_normalize_analyze_spending` uses `timedelta`; ensure `from datetime import timedelta` was added at top of `agent/llm.py`.
- `analyze_spending` SQL uses `func.date_trunc('week', ...)` which requires PostgreSQL; sqlite test fallbacks would fail here, but the project uses PostgreSQL only — confirm test DB URL is asyncpg.
- `_check_budget_alerts` is reused in batch; if a category does not exist, alerts are skipped (matches single-record behavior).
- If `test_post_feedback_records_row` regresses because `record_batch` now creates AI-flagged transactions and triggers feedback, verify that the test asserts feedback only on AI updates (not creates). Existing P9 hooks fire on `update_transaction` / `soft_delete_transaction` — `record_batch` only calls `create_transaction`, so no regression.

- [ ] **Step 3: Final commit if needed**

```
git add -A
git commit -m "fix: P10 follow-up adjustments after full-suite run"
```

---

## Self-Review

**Spec coverage (§4.2 + §4.4):**
- AGENT_TOOLS expansion → Task 1 (`record_batch`) + Task 2 (`query_transactions`) + Task 3 (`analyze_spending`) + Task 4 (`check_budget`)
- Function-Calling schema for `record_transaction` / `record_batch` → Task 1 schema matches §4.4 fields (`amount_yuan`, `direction`, `category`, `account_name`, `transaction_date`, `description`)
- `analyze_spending` parameters (`period`, `period_start`, `period_end`, `group_by`, `compare_with`) → Task 3 schema and normalizer
- post_record_actions parallel branch (check_budget → create_alert) → already covered by existing `_check_budget_alerts` reused in batch executor
- Golden set covers `record_batch`, `analyze_spending` scenarios → Task 6

**Out of P10 scope:**
- LangGraph migration (current sequential flow handles 8 tools fine)
- RAGAS scoring of `analyze_spending` answers (P12)
- Frontend (P11)
- Embedding-based category strengthening on record (`update_category_embedding`)

**Placeholder scan:** none.

**Type consistency:**
- `AgentIntent` `Literal` updated in Task 1 to include all 4 new names; tasks 2/3/4 reuse the same TypedDict.
- `normalize_intent` dispatcher in `llm.py` keeps single signature `(name: str, arguments: dict, original_message: str) -> AgentIntent`; new `_normalize_*` helpers match.
- `execute_intent` keeps single signature `(intent, team_id, user_id, db)`; new branches dispatch to private async functions with consistent return shape `dict[str, Any]` containing `status` + `message`.
- `_resolve_category`, `_resolve_account`, `_check_budget_alerts` reused — same signatures as in current executor.
- `finance_service.list_transactions` already supports `category_id` / `account_id` keyword filters (verified in `finance/service.py:147-162`); Task 2 reuses without modification.
- `finance_service.get_budget_usage` returns dict with `amount_fen`, `spent_fen`, `usage_ratio`, `period` (verified `finance/service.py:347-403`); Task 4 reuses field names directly.
- Few-shot examples in Task 5 keep English wording so `prompt_prefix_token_estimate()` stays byte-stable across requests.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-05-p10-agent-tool-expansion.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — same session, batched checkpoints.

Which approach?
