import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.embedding import _cache_key, get_embedding


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self.store[key] = value


def _floats_to_bytes(values: list[float]) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _bytes_to_floats(payload: bytes) -> list[float]:
    count = len(payload) // 4
    return list(struct.unpack(f"{count}f", payload))


@pytest.mark.asyncio
async def test_cache_miss_calls_openai_and_stores_bytes():
    redis = FakeRedis()
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    vector = await get_embedding("hello", redis=redis, client=fake_client)

    assert vector == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    fake_client.embeddings.create.assert_awaited_once()
    cached = await redis.get(_cache_key("hello"))
    assert _bytes_to_floats(cached) == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.asyncio
async def test_cache_hit_skips_openai():
    redis = FakeRedis()
    redis.store[_cache_key("hi")] = _floats_to_bytes([0.5, 0.6])
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock()

    vector = await get_embedding("hi", redis=redis, client=fake_client)

    assert vector == [pytest.approx(0.5), pytest.approx(0.6)]
    fake_client.embeddings.create.assert_not_awaited()
