import hashlib
import struct
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.redis import get_redis


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embed:{digest}"


def _floats_to_bytes(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _bytes_to_floats(payload: bytes) -> list[float]:
    count = len(payload) // 4
    return list(struct.unpack(f"{count}f", payload))


async def get_embedding(
    text: str,
    redis: Any = None,
    client: AsyncOpenAI | None = None,
) -> list[float]:
    redis = redis if redis is not None else await get_redis()
    key = _cache_key(text)
    cached = await redis.get(key)
    if cached:
        return _bytes_to_floats(cached)

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    vector = list(response.data[0].embedding)
    await redis.set(key, _floats_to_bytes(vector))
    return vector
