import uuid
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.rag.embedding import get_embedding
from app.rag.hybrid_search import _dense_search

HYDE_PROMPT = """假设你是一份财务政策文档，写一段可以回答下面问题的事实性内容。
限制在 80 字以内，直接陈述事实，不要解释。

问题: {query}"""


async def hyde_retrieve(
    query: str,
    team_id: uuid.UUID,
    db: AsyncSession,
    top_k: int = 20,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
    )
    hypothesis = (response.choices[0].message.content or query).strip()
    embedding = await get_embedding(hypothesis)
    return await _dense_search(embedding, team_id, db, top_k)
