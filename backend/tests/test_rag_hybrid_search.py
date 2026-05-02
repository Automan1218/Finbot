from unittest.mock import AsyncMock

import pytest

from app.rag.hybrid_search import hybrid_search, reciprocal_rank_fusion
from app.rag.ingest import ingest_document


def test_rrf_ranks_overlapping_doc_higher():
    dense = [{"id": "a", "chunk_text": "x"}, {"id": "b", "chunk_text": "y"}]
    sparse = [{"id": "b", "chunk_text": "y"}, {"id": "c", "chunk_text": "z"}]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)

    assert fused[0]["id"] == "b"
    assert {doc["id"] for doc in fused} == {"a", "b", "c"}


def test_rrf_handles_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []


@pytest.mark.asyncio
async def test_hybrid_search_returns_matching_chunks(db_session, finance_setup, monkeypatch):
    user, team, _ = finance_setup
    fake_embed = AsyncMock(side_effect=lambda text, **_: [0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)
    monkeypatch.setattr("app.rag.hybrid_search.get_embedding", fake_embed)

    await ingest_document(
        team_id=team.id,
        user_id=user.id,
        title="Travel Policy",
        source_type="policy",
        text="出差报销机票经济舱酒店500元以内餐饮150元",
        db=db_session,
        chunk_size=200,
        overlap=10,
    )

    results = await hybrid_search("出差", team.id, db_session, top_k=5)

    assert len(results) >= 1
    assert any("出差" in result["chunk_text"] for result in results)
