import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings

_PROMPT_TEMPLATE = (
    "Score each document chunk for relevance to the question on a 0-10 scale. "
    'Return only a JSON object shaped like {{"scores": [..]}} with exactly {count} scores.\n'
    "Question: {query}\n"
    "Documents: {docs}"
)


async def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    docs_text = json.dumps([c["chunk_text"] for c in candidates], ensure_ascii=False)
    prompt = _PROMPT_TEMPLATE.format(
        count=len(candidates), query=query, docs=docs_text
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        scores = json.loads(content).get("scores") or []
        if len(scores) != len(candidates):
            raise ValueError("score length mismatch")
    except (json.JSONDecodeError, ValueError, KeyError, AttributeError, TypeError):
        return candidates[:top_n]

    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]
