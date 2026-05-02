import hashlib
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.redis import get_redis

REWRITE_PROMPT = """将用户问题改写为更适合文档检索的形式。
扩展缩写，补全背景，保留核心意图。只输出改写后的问题，不要解释。

原问题: {query}
改写:"""

_TTL_SECONDS = 3600


def _cache_key(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"rewrite:{digest}"


async def rewrite_query(
    query: str,
    redis: Any = None,
    client: AsyncOpenAI | None = None,
) -> str:
    redis = redis if redis is not None else await get_redis()
    key = _cache_key(query)
    cached = await redis.get(key)
    if cached:
        return cached.decode("utf-8") if isinstance(cached, (bytes, bytearray)) else str(cached)

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
    )
    rewritten = (response.choices[0].message.content or query).strip()
    await redis.setex(key, _TTL_SECONDS, rewritten)
    return rewritten
