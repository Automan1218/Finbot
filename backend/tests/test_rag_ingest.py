from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.models.knowledge import Document, DocumentChunk
from app.rag.ingest import chunk_text, ingest_document


def test_chunk_text_splits_with_overlap():
    text = "abcdefghij" * 10
    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert len(chunks) >= 3
    assert chunks[0] == text[:30]
    assert chunks[1].startswith(text[25:30])


def test_chunk_text_returns_single_chunk_when_short():
    chunks = chunk_text("short", chunk_size=100, overlap=10)

    assert chunks == ["short"]


@pytest.mark.asyncio
async def test_ingest_document_persists_doc_and_chunks(db_session, finance_setup, monkeypatch):
    user, team, _ = finance_setup
    fake_embed = AsyncMock(side_effect=lambda text, **_: [0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    doc = await ingest_document(
        team_id=team.id,
        user_id=user.id,
        title="Policy",
        source_type="policy",
        text="abcdefghij" * 100,
        db=db_session,
        chunk_size=200,
        overlap=20,
    )

    assert isinstance(doc, Document)
    chunks = (
        await db_session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.doc_id == doc.id)
            .order_by(DocumentChunk.chunk_index)
        )
    ).scalars().all()
    assert len(chunks) >= 5
    assert all(len(chunk.embedding) == 1536 for chunk in chunks)
    assert chunks[0].chunk_index == 0


@pytest.mark.asyncio
async def test_ingest_document_rejects_empty_text(db_session, finance_setup):
    user, team, _ = finance_setup

    with pytest.raises(ValueError):
        await ingest_document(
            team_id=team.id,
            user_id=user.id,
            title="Empty",
            source_type="policy",
            text="   ",
            db=db_session,
        )
