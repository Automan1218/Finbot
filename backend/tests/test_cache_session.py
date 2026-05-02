import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.cache.session import (
    SESSION_TTL_SECONDS,
    build_session_window,
    summarize_messages,
)


class FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[bytes]] = {}
        self.ttls: dict[str, int] = {}

    async def lrange(self, key: str, start: int, stop: int):
        items = self.lists.get(key, [])
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    async def rpush(self, key: str, value):
        self.lists.setdefault(key, []).append(value if isinstance(value, bytes) else value.encode("utf-8"))

    async def delete(self, key: str):
        self.lists.pop(key, None)
        self.ttls.pop(key, None)

    async def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl


def _build_client(content: str) -> MagicMock:
    msg = MagicMock(content=content)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _push_messages(redis: FakeRedis, key: str, count: int) -> None:
    for i in range(count):
        msg = {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        redis.lists.setdefault(key, []).append(json.dumps(msg).encode("utf-8"))


@pytest.mark.asyncio
async def test_short_history_returns_messages_unchanged():
    redis = FakeRedis()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    key = f"session:{user_id}:{conv_id}"
    _push_messages(redis, key, 5)

    window = await build_session_window(redis, user_id, conv_id)

    assert len(window) == 5
    assert window[0]["content"] == "m0"


@pytest.mark.asyncio
async def test_long_history_returns_compressed_window(monkeypatch):
    redis = FakeRedis()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    key = f"session:{user_id}:{conv_id}"
    _push_messages(redis, key, 25)

    fake_summarize = AsyncMock(return_value="summary text")
    monkeypatch.setattr("app.cache.session.summarize_messages", fake_summarize)

    window = await build_session_window(
        redis, user_id, conv_id, max_history=10, compress_at=20
    )

    assert window[0]["role"] == "system"
    assert "summary text" in window[0]["content"]
    assert len(window) == 11
    assert window[-1]["content"] == "m24"
    assert redis.ttls[key] == SESSION_TTL_SECONDS


@pytest.mark.asyncio
async def test_summarize_messages_calls_llm_with_history():
    client = _build_client("compressed summary")
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]

    summary = await summarize_messages(messages, client=client)

    assert summary == "compressed summary"
    client.chat.completions.create.assert_awaited_once()
