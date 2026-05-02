import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.hybrid_search import hybrid_search
from app.rag.hyde import hyde_retrieve
from app.rag.reranker import rerank
from app.rag.rewrite import rewrite_query

_VAGUE_WORDS = (
    "怎么",
    "什么",
    "有没有",
    "能不能",
    "如何",
    "为什么",
    "啥",
    "多少",
    "吗",
)


def is_vague_query(query: str) -> bool:
    text = query.strip()
    if len(text) < 15:
        return True
    return any(word in text for word in _VAGUE_WORDS)


async def smart_retrieve(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if is_vague_query(query):
        candidates = await hyde_retrieve(query, team_id, db, top_k=top_k)
    else:
        rewritten = await rewrite_query(query)
        candidates = await hybrid_search(rewritten, team_id, db, top_k=top_k)

    if not candidates:
        return []
    return await rerank(query, candidates, top_n=top_n)
