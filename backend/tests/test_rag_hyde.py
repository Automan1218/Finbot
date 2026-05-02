import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.hyde import hyde_retrieve


def _build_client(content: str) -> MagicMock:
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_hyde_calls_llm_then_dense_search(monkeypatch):
    client = _build_client("出差报销：机票经济舱，酒店每日500元以内。")
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    fake_dense = AsyncMock(return_value=[{"id": "x", "chunk_text": "policy"}])
    monkeypatch.setattr("app.rag.hyde.get_embedding", fake_embed)
    monkeypatch.setattr("app.rag.hyde._dense_search", fake_dense)

    fake_db = MagicMock()
    team_id = uuid.uuid4()
    results = await hyde_retrieve("出差有什么限制", team_id, fake_db, client=client)

    assert results == [{"id": "x", "chunk_text": "policy"}]
    client.chat.completions.create.assert_awaited_once()
    fake_embed.assert_awaited_once()
    fake_dense.assert_awaited_once()
