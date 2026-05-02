from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.rewrite import _cache_key, rewrite_query


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    async def get(self, key: str):
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value.encode("utf-8")


def _build_client(content: str) -> MagicMock:
    message = MagicMock(content=content)
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_rewrite_caches_result():
    redis = FakeRedis()
    client = _build_client("差旅费报销 流程 审批 金额阈值")

    rewritten = await rewrite_query("这笔要审批吗", redis=redis, client=client)
    cached = await rewrite_query("这笔要审批吗", redis=redis, client=client)

    assert "审批" in rewritten
    assert cached == rewritten
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_rewrite_cache_hit_skips_llm():
    redis = FakeRedis()
    redis.store[_cache_key("hi")] = b"already-rewritten"
    client = _build_client("should-not-be-used")

    result = await rewrite_query("hi", redis=redis, client=client)

    assert result == "already-rewritten"
    client.chat.completions.create.assert_not_awaited()
