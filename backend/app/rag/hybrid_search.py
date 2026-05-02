import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embedding import get_embedding


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for rank, doc in enumerate(dense):
        doc_id = str(doc["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc in enumerate(sparse):
        doc_id = str(doc["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    doc_map: dict[str, dict[str, Any]] = {}
    for doc in dense + sparse:
        doc_map.setdefault(str(doc["id"]), doc)
    return [doc_map[doc_id] for doc_id in sorted(scores, key=scores.get, reverse=True)]


async def _dense_search(
    embedding: list[float],
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int,
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT dc.id, dc.chunk_text,
               1 - (dc.embedding <=> CAST(:embedding AS vector)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE d.team_id = :team_id AND dc.embedding IS NOT NULL
        ORDER BY dc.embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
        """
    )
    embedding_literal = "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"
    result = await db.execute(
        sql,
        {"embedding": embedding_literal, "team_id": str(team_id), "top_k": top_k},
    )
    return [
        {"id": row.id, "chunk_text": row.chunk_text, "score": float(row.score)}
        for row in result
    ]


async def _sparse_search(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int,
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT dc.id, dc.chunk_text,
               ts_rank(dc.tsv, plainto_tsquery('simple', :query)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE d.team_id = :team_id
          AND dc.tsv @@ plainto_tsquery('simple', :query)
        ORDER BY score DESC
        LIMIT :top_k
        """
    )
    result = await db.execute(
        sql,
        {"query": query, "team_id": str(team_id), "top_k": top_k},
    )
    return [
        {"id": row.id, "chunk_text": row.chunk_text, "score": float(row.score)}
        for row in result
    ]


async def hybrid_search(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    embedding = await get_embedding(query)
    dense = await _dense_search(embedding, team_id, db, top_k)
    sparse = await _sparse_search(query, team_id, db, top_k)
    return reciprocal_rank_fusion(dense, sparse)
