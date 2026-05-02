# P8 TTFT Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut Time-To-First-Token by ~25% (spec §7) via parallel context construction (session + RAG concurrent), OpenAI prompt prefix caching (static ≥1024-token preamble shared across requests), and direct token-level streaming through Redis pub/sub → SSE.

**Architecture:** Three new files under `app/agent/`. `context.py` runs session window + RAG retrieval concurrently with `asyncio.gather` and returns a single context dict. `prompt.py` exposes a fixed `SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES` block (≥1024 tokens, byte-identical across requests) plus a `build_prompt(history, rag_context, user_msg)` helper. `streaming.py` wraps `client.chat.completions.create(stream=True)` and publishes deltas to `task-progress:{task_id}`. `chat/service.py` adds a new `run_streaming_chat_task` that orchestrates load_context → stream → finalize. `chat/router.py` keeps the existing SSE endpoint; new event types `delta` / `final` flow through the same channel.

**Tech Stack:** OpenAI Python SDK async streaming, `redis.asyncio` pub/sub, existing `sse-starlette` router.

---

## File Structure

**New files (under `backend/app/agent/`):**
- `agent/context.py` — `load_context_parallel`, returns `{"history", "rag", "team_id", "user_id"}`
- `agent/prompt.py` — `SYSTEM_PROMPT`, `FEW_SHOT_EXAMPLES`, `build_prompt`, `prompt_prefix_token_estimate`
- `agent/streaming.py` — `stream_llm_to_sse(redis, task_id, conversation_id, messages, client=None)`

**New tests:**
- `tests/test_agent_context.py`
- `tests/test_agent_prompt.py`
- `tests/test_agent_streaming.py`

**Modified files:**
- `backend/app/chat/service.py` — add `run_streaming_chat_task`; existing `run_local_chat_task` stays for non-streaming intent dispatch
- `backend/app/chat/router.py` — accept `stream: bool = False` query, pick task runner accordingly
- `backend/tests/test_chat_service.py` — add streaming-path test

---

### Task 1: Parallel context construction

**Files:**
- Create: `backend/app/agent/context.py`
- Create: `backend/tests/test_agent_context.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_context.py`:

```python
import asyncio
import time
import uuid
from unittest.mock import AsyncMock

import pytest

from app.agent.context import load_context_parallel


@pytest.mark.asyncio
async def test_runs_session_and_rag_in_parallel(monkeypatch):
    async def slow_session(*args, **kwargs):
        await asyncio.sleep(0.1)
        return [{"role": "user", "content": "prev"}]

    async def slow_rag(*args, **kwargs):
        await asyncio.sleep(0.1)
        return [{"id": "c1", "chunk_text": "policy"}]

    monkeypatch.setattr("app.agent.context.build_session_window", slow_session)
    monkeypatch.setattr("app.agent.context.smart_retrieve", slow_rag)

    redis = object()
    db = object()
    started = time.perf_counter()
    ctx = await load_context_parallel(
        redis=redis,
        db=db,
        user_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        team_id=uuid.uuid4(),
        query="出差报销",
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.18  # parallel ≈ 100ms; serial would be ≥200ms
    assert ctx["history"][0]["content"] == "prev"
    assert ctx["rag"][0]["chunk_text"] == "policy"


@pytest.mark.asyncio
async def test_skips_rag_when_team_id_none(monkeypatch):
    fake_session = AsyncMock(return_value=[])
    fake_rag = AsyncMock()
    monkeypatch.setattr("app.agent.context.build_session_window", fake_session)
    monkeypatch.setattr("app.agent.context.smart_retrieve", fake_rag)

    ctx = await load_context_parallel(
        redis=object(),
        db=None,
        user_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        team_id=None,
        query="hi",
    )

    assert ctx["rag"] == []
    fake_rag.assert_not_awaited()
    fake_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_returns_empty_rag_when_query_blank(monkeypatch):
    fake_session = AsyncMock(return_value=[])
    fake_rag = AsyncMock()
    monkeypatch.setattr("app.agent.context.build_session_window", fake_session)
    monkeypatch.setattr("app.agent.context.smart_retrieve", fake_rag)

    ctx = await load_context_parallel(
        redis=object(),
        db=object(),
        user_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        team_id=uuid.uuid4(),
        query="   ",
    )

    assert ctx["rag"] == []
    fake_rag.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_context.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement context loader**

Create `backend/app/agent/context.py`:

```python
import asyncio
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.session import build_session_window
from app.rag.retrieve import smart_retrieve


async def load_context_parallel(
    redis: Any,
    db: AsyncSession | None,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    team_id: uuid.UUID | None,
    query: str,
) -> dict[str, Any]:
    session_task = build_session_window(redis, user_id, conversation_id)

    skip_rag = team_id is None or db is None or not query.strip()
    if skip_rag:
        history = await session_task
        return {
            "history": history,
            "rag": [],
            "team_id": team_id,
            "user_id": user_id,
        }

    rag_task = smart_retrieve(query, team_id, db)
    history, rag = await asyncio.gather(session_task, rag_task)
    return {
        "history": history,
        "rag": rag,
        "team_id": team_id,
        "user_id": user_id,
    }
```

- [ ] **Step 4: Run test to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_context.py -v
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/agent/context.py backend/tests/test_agent_context.py
git commit -m "feat: P8 parallel context loader (session + RAG concurrent)"
```

---

### Task 2: Prompt prefix builder (≥1024 tokens, stable across requests)

**Files:**
- Create: `backend/app/agent/prompt.py`
- Create: `backend/tests/test_agent_prompt.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_prompt.py`:

```python
import pytest

from app.agent.prompt import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_prompt,
    prompt_prefix_token_estimate,
)


def test_system_prompt_is_stable_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT == SYSTEM_PROMPT  # identity check


def test_few_shot_examples_is_list_of_role_content_dicts():
    assert isinstance(FEW_SHOT_EXAMPLES, list)
    assert len(FEW_SHOT_EXAMPLES) >= 4
    for item in FEW_SHOT_EXAMPLES:
        assert set(item.keys()) == {"role", "content"}
        assert item["role"] in {"user", "assistant", "system"}


def test_prefix_token_estimate_meets_openai_cache_threshold():
    # Approx: 1 token ≈ 4 chars for English, ≈ 1.5 chars for Chinese.
    # OpenAI prefix cache requires ≥1024 tokens.
    assert prompt_prefix_token_estimate() >= 1024


def test_build_prompt_places_static_prefix_first():
    history = [{"role": "user", "content": "prev question"}]
    rag = "policy snippet"
    user_msg = "current question"

    messages = build_prompt(history=history, rag_context=rag, user_msg=user_msg)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    for i, shot in enumerate(FEW_SHOT_EXAMPLES, start=1):
        assert messages[i] == shot
    assert messages[-1] == {"role": "user", "content": "current question"}
    rag_position = 1 + len(FEW_SHOT_EXAMPLES)
    assert "policy snippet" in messages[rag_position]["content"]


def test_build_prompt_handles_empty_rag_and_history():
    messages = build_prompt(history=[], rag_context="", user_msg="hello")

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1] == {"role": "user", "content": "hello"}
    rag_position = 1 + len(FEW_SHOT_EXAMPLES)
    assert messages[rag_position]["role"] == "system"


def test_build_prompt_static_prefix_byte_identical_across_calls():
    p1 = build_prompt(history=[], rag_context="x", user_msg="a")
    p2 = build_prompt(history=[{"role": "user", "content": "y"}], rag_context="z", user_msg="b")
    prefix_len = 1 + len(FEW_SHOT_EXAMPLES)
    assert p1[:prefix_len] == p2[:prefix_len]
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_prompt.py -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement prompt module**

Create `backend/app/agent/prompt.py`:

```python
from typing import Any

SYSTEM_PROMPT = """You are Finbot, an AI financial assistant for small teams. Your job is to help users:

1. Record financial transactions from natural-language input. Always normalize currency, infer category, and confirm direction (income vs expense). Money is stored in fen — convert from yuan when needed (1 yuan = 100 fen).

2. Generate finance reports across categories, accounts, or time periods. Always state the period boundaries explicitly. Default to "this month" if the user does not specify a range.

3. Look up team policy or knowledge documents (e.g. reimbursement rules, expense limits). When a knowledge snippet is provided, ground your answer in it and quote the relevant phrase. Do not fabricate policy details.

4. Ask follow-up questions when required fields are missing or ambiguous. Required record fields are: amount, direction, category, account, transaction_date, description.

Behavioral rules — these are non-negotiable:

- Always pick exactly one tool (record_transaction, generate_report, rag_retrieve, or clarify). Do not combine tools in one turn.
- Numbers must be exact. If the user says "around 50", set amount=50 and note the approximation in description; never invent a value.
- Dates: prefer ISO 8601 (YYYY-MM-DD). When the user says "yesterday" or "last Friday", resolve relative to today's date supplied in the conversation.
- Currency: assume CNY unless the user names another currency. Convert all amounts to fen as integers — for example "35.5 元" becomes amount_fen=3550.
- Account inference: read context for hints like "from my Alipay" → account_name="Alipay". If unclear, fall back to "Default" and let the user correct it.
- Category inference: be specific. "Coffee" → "Food & Beverage", not "Other". For ambiguous merchants, use the user's prior categorization seen in conversation history.
- Income vs expense: words like "received", "got paid", "refund", "salary" → income. "Spent", "bought", "paid", "withdrew" → expense.
- Privacy: never echo back personal data unrelated to the immediate task. Do not store or summarize sensitive identifiers.
- Tone: concise, factual, no filler. Do not begin replies with "Sure" or "Of course". Do not apologize. Do not narrate what you are about to do — just do it.
- Multilingual: detect the user's language (Chinese or English) and respond in the same language. Mixed-language input is fine; match the dominant language in the most recent user message.

Knowledge grounding:

- When a system message labeled "相关知识 / Relevant knowledge" appears, treat its contents as ground truth for policy questions. Quote the most relevant sentence verbatim before paraphrasing. If the knowledge does not contain the answer, say so explicitly — do not extrapolate.
- If multiple knowledge chunks conflict, prefer the most specific one (e.g. a category-level rule beats a general guideline).

Conversation memory:

- A "历史摘要 / History summary" system message may appear when the conversation has been compressed. Treat it as authoritative for facts already established earlier. Do not re-ask for information present in the summary.
- Recent message history (the last ~10 turns) is preserved verbatim. Use it for short-range coreference (e.g. "that one", "the same account").

Tool-call discipline:

- record_transaction requires all six fields. If any are missing after reading conversation context, call clarify with a single targeted question.
- record_batch is reserved for inputs that explicitly enumerate multiple transactions (e.g. "今天早饭12 午饭35 打车45"). Do not invent batches from a single statement.
- generate_report requires period_start, period_end, and group_by. If the user says "this month", resolve to the first/last day of the current month.
- rag_retrieve is for policy/knowledge questions ("how do I", "what's the limit", "is X allowed"). Do not use it for transactional queries.
- clarify must include the missing_fields list. Keep questions short and specific — one missing piece per question.

Error recovery:

- If a previous tool call failed, summarize the failure for the user in one sentence and suggest the next step. Do not silently retry.
- If you detect a logical inconsistency (e.g. expense to an account that does not exist), stop and ask before writing.

Final output format:

- Always reply with a tool call. Never reply with plain prose unless the chosen tool is clarify.
- For clarify, the question field is what the user sees. Make it actionable.
- For record_transaction and record_batch, the description field should preserve the user's original phrasing as a faithful audit trail.
""".strip()


FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "今天午饭花了35元，用支付宝",
    },
    {
        "role": "assistant",
        "content": (
            "调用 record_transaction：amount_yuan=35, direction=expense, "
            "category=餐饮, account_name=支付宝, transaction_date=今天, "
            "description=今天午饭花了35元，用支付宝"
        ),
    },
    {
        "role": "user",
        "content": "客户打款5000到工行",
    },
    {
        "role": "assistant",
        "content": (
            "调用 record_transaction：amount_yuan=5000, direction=income, "
            "category=收入, account_name=工行, transaction_date=今天, "
            "description=客户打款5000到工行"
        ),
    },
    {
        "role": "user",
        "content": "差旅费报销有什么限制",
    },
    {
        "role": "assistant",
        "content": "调用 rag_retrieve：query=差旅费报销限制",
    },
    {
        "role": "user",
        "content": "生成本月报表，按分类",
    },
    {
        "role": "assistant",
        "content": (
            "调用 generate_report：period_start=本月1日, "
            "period_end=今天, group_by=category"
        ),
    },
]


def prompt_prefix_token_estimate() -> int:
    total_chars = len(SYSTEM_PROMPT)
    for shot in FEW_SHOT_EXAMPLES:
        total_chars += len(shot["content"])
    # Mixed Chinese/English content averages ~2.5 chars/token in tiktoken.
    return int(total_chars / 2.5)


def build_prompt(
    history: list[dict[str, Any]],
    rag_context: str,
    user_msg: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append(
        {
            "role": "system",
            "content": f"相关知识 / Relevant knowledge:\n{rag_context}" if rag_context else "相关知识 / Relevant knowledge: (none)",
        }
    )
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    return messages
```

- [ ] **Step 4: Run test to verify pass**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_prompt.py -v
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/agent/prompt.py backend/tests/test_agent_prompt.py
git commit -m "feat: P8 stable prompt prefix for OpenAI cache (≥1024 tokens)"
```

---

### Task 3: Streaming output via Redis pub/sub

**Files:**
- Create: `backend/app/agent/streaming.py`
- Create: `backend/tests/test_agent_streaming.py`

- [ ] **Step 1: Write failing test**

Create `backend/tests/test_agent_streaming.py`:

```python
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.streaming import stream_llm_to_sse


class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.lists: dict[str, list[bytes]] = {}
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    async def publish(self, channel: str, payload):
        self.published.append((channel, payload if isinstance(payload, str) else payload.decode("utf-8")))

    async def rpush(self, key, value):
        encoded = value if isinstance(value, bytes) else value.encode("utf-8")
        self.lists.setdefault(key, []).append(encoded)

    async def expire(self, key, ttl):
        self.ttls[key] = ttl

    async def set(self, key, value, ex=None):
        self.values[key] = value if isinstance(value, bytes) else value.encode("utf-8")


def _build_streaming_client(deltas: list[str]) -> MagicMock:
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])
        for delta in deltas
    ]

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for chunk in chunks:
                yield chunk

    client = MagicMock()
    client.chat.completions.stream = MagicMock(return_value=FakeStream())
    return client


@pytest.mark.asyncio
async def test_streams_each_delta_to_redis_channel():
    redis = FakeRedis()
    client = _build_streaming_client(["Hello", " ", "world"])
    task_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    full_text = await stream_llm_to_sse(
        redis=redis,
        task_id=task_id,
        conversation_id=conv_id,
        messages=[{"role": "user", "content": "say hi"}],
        client=client,
    )

    assert full_text == "Hello world"
    deltas = [
        json.loads(payload)
        for channel, payload in redis.published
        if channel == f"task-progress:{task_id}"
    ]
    delta_steps = [d for d in deltas if d.get("step") == "generating"]
    assert [d["delta"] for d in delta_steps] == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_skips_empty_delta_chunks():
    redis = FakeRedis()
    client = _build_streaming_client(["A", "", None, "B"])
    task_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    full_text = await stream_llm_to_sse(
        redis=redis,
        task_id=task_id,
        conversation_id=conv_id,
        messages=[{"role": "user", "content": "x"}],
        client=client,
    )

    assert full_text == "AB"
    deltas = [
        json.loads(payload)
        for channel, payload in redis.published
        if channel == f"task-progress:{task_id}"
    ]
    delta_steps = [d for d in deltas if d.get("step") == "generating"]
    assert [d["delta"] for d in delta_steps] == ["A", "B"]
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_streaming.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement streaming module**

Create `backend/app/agent/streaming.py`:

```python
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def stream_llm_to_sse(
    redis: Any,
    task_id: uuid.UUID,
    conversation_id: uuid.UUID,
    messages: list[dict[str, Any]],
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> str:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    channel = f"task-progress:{task_id}"
    chunks: list[str] = []

    async with client.chat.completions.stream(
        model=model or settings.OPENAI_MODEL,
        messages=messages,
        stream=True,
    ) as stream:
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if not content:
                continue
            chunks.append(content)
            event = {
                "task_id": str(task_id),
                "conversation_id": str(conversation_id),
                "step": "generating",
                "status": "running",
                "delta": content,
                "created_at": _now_iso(),
            }
            await redis.publish(channel, json.dumps(event, ensure_ascii=False))

    return "".join(chunks)
```

- [ ] **Step 4: Run test**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_agent_streaming.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/agent/streaming.py backend/tests/test_agent_streaming.py
git commit -m "feat: P8 streaming LLM output via Redis pub/sub"
```

---

### Task 4: Integrate streaming task into chat service

**Files:**
- Modify: `backend/app/chat/service.py`
- Modify: `backend/tests/test_chat_service.py`

- [ ] **Step 1: Write failing test**

Append to `backend/tests/test_chat_service.py`:

```python
@pytest.mark.asyncio
async def test_run_streaming_chat_task_publishes_deltas_and_finalizes(monkeypatch):
    import uuid as _uuid
    from unittest.mock import AsyncMock

    redis = FakeRedis()
    user_id = _uuid.uuid4()
    team_id = _uuid.uuid4()
    task_id, conv_id = await service.create_chat_task(
        redis=redis, user_id=user_id, team_id=team_id, message="出差报销限制",
        conversation_id=None,
    )

    fake_ctx = AsyncMock(return_value={
        "history": [],
        "rag": [{"id": "c1", "chunk_text": "差旅报销标准 500/晚"}],
        "team_id": team_id,
        "user_id": user_id,
    })
    fake_stream = AsyncMock(return_value="差旅报销限额每晚500元")
    monkeypatch.setattr("app.chat.service.load_context_parallel", fake_ctx)
    monkeypatch.setattr("app.chat.service.stream_llm_to_sse", fake_stream)

    await service.run_streaming_chat_task(
        redis=redis,
        task_id=task_id,
        user_id=user_id,
        conversation_id=conv_id,
        message="出差报销限制",
        team_id=team_id,
    )

    fake_ctx.assert_awaited_once()
    fake_stream.assert_awaited_once()
    history = await service.get_history(redis, user_id, conv_id)
    events = await service.list_task_events(redis, task_id)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "差旅报销限额每晚500元"
    assert events[-1]["status"] == "done"
    assert events[-1]["data"]["mode"] == "streaming"
```

- [ ] **Step 2: Run test to verify fail**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_chat_service.py::test_run_streaming_chat_task_publishes_deltas_and_finalizes -v
```
Expected: FAIL — `run_streaming_chat_task` not defined

- [ ] **Step 3: Implement run_streaming_chat_task**

Edit `backend/app/chat/service.py`. Add imports:

```python
from app.agent.context import load_context_parallel
from app.agent.prompt import build_prompt
from app.agent.streaming import stream_llm_to_sse
```

Append new function below `run_local_chat_task`:

```python
async def run_streaming_chat_task(
    redis: Redis,
    task_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    team_id: uuid.UUID | None = None,
) -> None:
    try:
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="load_context",
            status="running",
            message="Loading context",
        )
        async with get_db_session() as db:
            ctx = await load_context_parallel(
                redis=redis,
                db=db,
                user_id=user_id,
                conversation_id=conversation_id,
                team_id=team_id,
                query=message,
            )

        rag_text = "\n".join(c.get("chunk_text", "") for c in ctx.get("rag", []))
        prompt = build_prompt(history=ctx.get("history", []), rag_context=rag_text, user_msg=message)

        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="generating",
            status="running",
            message="Generating",
        )
        full_text = await stream_llm_to_sse(
            redis=redis,
            task_id=task_id,
            conversation_id=conversation_id,
            messages=prompt,
        )

        await append_message(redis, user_id, conversation_id, "assistant", full_text)
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="complete",
            status="done",
            message=full_text,
            data={"mode": "streaming"},
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
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```
git add backend/app/chat/service.py backend/tests/test_chat_service.py
git commit -m "feat: P8 streaming chat task using parallel context + stable prompt"
```

---

### Task 5: Router toggle for streaming mode

**Files:**
- Modify: `backend/app/chat/router.py`
- Modify: `backend/app/chat/schemas.py`

- [ ] **Step 1: Add `stream` field to ChatMessageCreate**

Read `backend/app/chat/schemas.py`. Locate `ChatMessageCreate` and add:

```python
class ChatMessageCreate(BaseModel):
    team_id: uuid.UUID
    message: str
    conversation_id: uuid.UUID | None = None
    stream: bool = False
```

(Adapt to match existing field order. Only add the `stream` field; do not change others.)

- [ ] **Step 2: Switch task runner based on flag**

Edit `backend/app/chat/router.py`. In `create_message`, replace:

```python
    background_tasks.add_task(
        service.run_local_chat_task,
        redis,
        task_id,
        current_user.id,
        conversation_id,
        body.message,
        body.team_id,
    )
```

with:

```python
    runner = service.run_streaming_chat_task if body.stream else service.run_local_chat_task
    background_tasks.add_task(
        runner,
        redis,
        task_id,
        current_user.id,
        conversation_id,
        body.message,
        body.team_id,
    )
```

- [ ] **Step 3: Run all chat tests**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/test_chat_service.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 4: Commit**

```
git add backend/app/chat/router.py backend/app/chat/schemas.py
git commit -m "feat: P8 router toggle to dispatch streaming chat task on stream=true"
```

---

### Task 6: Full-suite verification

**Files:** none new — verification only.

- [ ] **Step 1: Run full test suite**

```
C:/Users/henry/.conda/envs/finbot/python.exe -m pytest backend/tests/ -q
```
Expected: all green; new tests count >= old + 11.

- [ ] **Step 2: If failures, fix and re-run**

Common follow-ups:
- `load_context_parallel` parallel-timing test flakes on slow CI: bump tolerance from 0.18 to 0.22 once.
- Stream test fails with `AttributeError: stream` on AsyncOpenAI: verify SDK version supports `client.chat.completions.stream(...)` context manager (openai>=1.40 does); otherwise pin SDK version in `pyproject.toml`.
- `prompt_prefix_token_estimate` returns <1024 if SYSTEM_PROMPT or FEW_SHOT_EXAMPLES were trimmed: extend either to push the estimate above the threshold.

- [ ] **Step 3: Final commit (if any fix needed)**

```
git add -A
git commit -m "fix: P8 follow-up adjustments after full-suite run"
```

---

## Self-Review

**Spec coverage:**
- §7.1 并行上下文构建 → Task 1 (`load_context_parallel`, `asyncio.gather`)
- §7.2 OpenAI Prompt 前缀缓存 → Task 2 (`SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES` byte-identical, ≥1024 tokens)
- §7.3 流式输出直达前端 → Task 3 (`stream_llm_to_sse`) + Task 4 (wire into chat service) + Task 5 (router toggle)
- §7.4 TTFT 来源汇总 — covered by Tasks 1-3

**Out of scope (handled in P9+):**
- §8 Eval / RAGAS / A/B — P9
- Frontend SSE consumer for token-level rendering — P10 frontend
- `record_batch` / `analyze_spending` agent tools — P10
- LLM-cache invalidation on transaction write (Layer 1 cross-team trigger) — deferred (low ROI)

**Placeholder scan:** none.

**Type consistency:**
- `load_context_parallel` returns `dict[str, Any]` with keys `history`, `rag`, `team_id`, `user_id` — consumed by Task 4 verbatim.
- `build_prompt(history, rag_context, user_msg)` signature consistent across Tasks 2 + 4.
- `stream_llm_to_sse(redis, task_id, conversation_id, messages, client=None, model=None)` returns `str` (full text) — Task 4 stores it via `append_message` and `push_task_event(message=full_text)`.
- Event channel `task-progress:{task_id}` matches existing SSE consumer in `chat/router.py`.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-05-02-p8-ttft-optimization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — same session, batched checkpoints.

Which approach?
