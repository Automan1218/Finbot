import uuid
from unittest.mock import AsyncMock

import pytest

from app.rag.retrieve import is_vague_query, smart_retrieve


def test_is_vague_short_query():
    assert is_vague_query("出差") is True


def test_is_vague_question_words():
    assert is_vague_query("出差报销有什么限制") is True


def test_is_vague_specific_query():
    assert is_vague_query("差旅费报销标准2026年最新版本说明") is False


@pytest.mark.asyncio
async def test_smart_retrieve_uses_hyde_for_vague(monkeypatch):
    fake_hyde = AsyncMock(return_value=[{"id": "h", "chunk_text": "hyde"}])
    fake_hybrid = AsyncMock(return_value=[{"id": "x", "chunk_text": "hybrid"}])
    fake_rewrite = AsyncMock(return_value="should-not-be-called")
    fake_rerank = AsyncMock(side_effect=lambda q, c, top_n, **_: c[:top_n])
    monkeypatch.setattr("app.rag.retrieve.hyde_retrieve", fake_hyde)
    monkeypatch.setattr("app.rag.retrieve.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.rag.retrieve.rewrite_query", fake_rewrite)
    monkeypatch.setattr("app.rag.retrieve.rerank", fake_rerank)

    results = await smart_retrieve("出差有啥限制", uuid.uuid4(), db=None, top_n=3)

    assert results == [{"id": "h", "chunk_text": "hyde"}]
    fake_hyde.assert_awaited_once()
    fake_hybrid.assert_not_awaited()
    fake_rewrite.assert_not_awaited()


@pytest.mark.asyncio
async def test_smart_retrieve_uses_rewrite_hybrid_for_specific(monkeypatch):
    fake_hyde = AsyncMock()
    fake_hybrid = AsyncMock(return_value=[{"id": "x", "chunk_text": "hybrid"}])
    fake_rewrite = AsyncMock(return_value="rewritten")
    fake_rerank = AsyncMock(side_effect=lambda q, c, top_n, **_: c[:top_n])
    monkeypatch.setattr("app.rag.retrieve.hyde_retrieve", fake_hyde)
    monkeypatch.setattr("app.rag.retrieve.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.rag.retrieve.rewrite_query", fake_rewrite)
    monkeypatch.setattr("app.rag.retrieve.rerank", fake_rerank)

    results = await smart_retrieve(
        "差旅费报销标准2026最新版本", uuid.uuid4(), db=None, top_n=5
    )

    assert results == [{"id": "x", "chunk_text": "hybrid"}]
    fake_rewrite.assert_awaited_once()
    assert fake_rewrite.await_args.args[0] == "差旅费报销标准2026最新版本"
    fake_hybrid.assert_awaited_once()
    assert fake_hybrid.await_args.args[0] == "rewritten"
    fake_hyde.assert_not_awaited()
