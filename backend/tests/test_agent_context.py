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

    started = time.perf_counter()
    ctx = await load_context_parallel(
        redis=object(),
        db=object(),
        user_id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        team_id=uuid.uuid4(),
        query="travel reimbursement",
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.18
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
