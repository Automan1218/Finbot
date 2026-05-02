import json
import uuid
from unittest.mock import MagicMock

import pytest

from app.agent.streaming import stream_llm_to_sse


class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.lists: dict[str, list[bytes]] = {}
        self.values: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}

    async def publish(self, channel: str, payload):
        decoded = payload if isinstance(payload, str) else payload.decode("utf-8")
        self.published.append((channel, decoded))

    async def rpush(self, key, value):
        encoded = value if isinstance(value, bytes) else value.encode("utf-8")
        self.lists.setdefault(key, []).append(encoded)

    async def expire(self, key, ttl):
        self.ttls[key] = ttl

    async def set(self, key, value, ex=None):
        self.values[key] = value if isinstance(value, bytes) else value.encode("utf-8")


def _build_streaming_client(deltas: list[str | None]) -> MagicMock:
    chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=delta))])
        for delta in deltas
    ]

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for chunk in chunks:
                yield chunk

    client = MagicMock()
    client.chat.completions.stream = MagicMock(return_value=FakeStream())
    return client


@pytest.mark.asyncio
async def test_streams_each_delta_to_redis_channel():
    redis = FakeRedis()
    client = _build_streaming_client(["Hello", " ", "world"])
    task_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    full_text = await stream_llm_to_sse(
        redis=redis,
        task_id=task_id,
        conversation_id=conv_id,
        messages=[{"role": "user", "content": "say hi"}],
        client=client,
    )

    assert full_text == "Hello world"
    deltas = [
        json.loads(payload)
        for channel, payload in redis.published
        if channel == f"task-progress:{task_id}"
    ]
    delta_steps = [d for d in deltas if d.get("step") == "generating"]
    assert [d["delta"] for d in delta_steps] == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_skips_empty_delta_chunks():
    redis = FakeRedis()
    client = _build_streaming_client(["A", "", None, "B"])
    task_id = uuid.uuid4()
    conv_id = uuid.uuid4()

    full_text = await stream_llm_to_sse(
        redis=redis,
        task_id=task_id,
        conversation_id=conv_id,
        messages=[{"role": "user", "content": "x"}],
        client=client,
    )

    assert full_text == "AB"
    deltas = [
        json.loads(payload)
        for channel, payload in redis.published
        if channel == f"task-progress:{task_id}"
    ]
    delta_steps = [d for d in deltas if d.get("step") == "generating"]
    assert [d["delta"] for d in delta_steps] == ["A", "B"]


@pytest.mark.asyncio
async def test_persists_delta_events_for_existing_sse_polling_path():
    redis = FakeRedis()
    client = _build_streaming_client(["A"])
    task_id = uuid.uuid4()

    await stream_llm_to_sse(
        redis=redis,
        task_id=task_id,
        conversation_id=uuid.uuid4(),
        messages=[{"role": "user", "content": "x"}],
        client=client,
    )

    events_key = f"chat-task-events:{task_id}"
    assert json.loads(redis.lists[events_key][0])["delta"] == "A"
