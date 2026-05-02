import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.reranker import rerank


def _build_client(scores: list[float]) -> MagicMock:
    payload = json.dumps({"scores": scores})
    message = MagicMock(content=payload)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_rerank_returns_top_n_in_order():
    candidates = [
        {"id": "a", "chunk_text": "alpha"},
        {"id": "b", "chunk_text": "beta"},
        {"id": "c", "chunk_text": "gamma"},
    ]
    client = _build_client([1.0, 9.5, 4.0])

    ranked = await rerank("query", candidates, top_n=2, client=client)

    assert [doc["id"] for doc in ranked] == ["b", "c"]


@pytest.mark.asyncio
async def test_rerank_empty_returns_empty():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()

    result = await rerank("q", [], top_n=5, client=client)

    assert result == []
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_rerank_falls_back_when_response_invalid():
    candidates = [{"id": "a", "chunk_text": "alpha"}, {"id": "b", "chunk_text": "beta"}]
    message = MagicMock(content="not-json")
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    result = await rerank("q", candidates, top_n=2, client=client)

    assert result == candidates[:2]
