import uuid

import pytest

from app.chat import service


class FakeRedis:
    def __init__(self):
        self.lists = {}
        self.values = {}
        self.deleted = []
        self.published = []

    async def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    async def lrange(self, key, start, stop):
        items = self.lists.get(key, [])
        if stop == -1:
            return items[start:]
        return items[start : stop + 1]

    async def expire(self, key, seconds):
        return True

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def publish(self, channel, payload):
        self.published.append((channel, payload))

    async def delete(self, key):
        self.deleted.append(key)
        self.lists.pop(key, None)


@pytest.mark.asyncio
async def test_create_chat_task_stores_user_message_and_event():
    redis = FakeRedis()
    user_id = uuid.uuid4()
    team_id = uuid.uuid4()

    task_id, conversation_id = await service.create_chat_task(
        redis=redis,
        user_id=user_id,
        team_id=team_id,
        message="今天午饭花了 35 元",
        conversation_id=None,
    )

    history = await service.get_history(redis, user_id, conversation_id)
    events = await service.list_task_events(redis, task_id)
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "今天午饭花了 35 元"
    assert events[0]["status"] == "queued"
    assert events[0]["data"]["team_id"] == str(team_id)


@pytest.mark.asyncio
async def test_run_local_chat_task_adds_assistant_reply_and_done_event():
    redis = FakeRedis()
    user_id = uuid.uuid4()
    team_id = uuid.uuid4()
    task_id, conversation_id = await service.create_chat_task(
        redis=redis,
        user_id=user_id,
        team_id=team_id,
        message="生成本月报表",
        conversation_id=None,
    )

    await service.run_local_chat_task(
        redis=redis,
        task_id=task_id,
        user_id=user_id,
        conversation_id=conversation_id,
        message="生成本月报表",
    )

    history = await service.get_history(redis, user_id, conversation_id)
    events = await service.list_task_events(redis, task_id)
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert events[-1]["status"] == "done"
    assert "报表" in events[-1]["message"]


@pytest.mark.asyncio
async def test_run_local_chat_task_uses_cached_llm_response(monkeypatch):
    import json
    from unittest.mock import AsyncMock

    fake_redis = AsyncMock()
    cached_payload = {
        "intent": {"name": "clarify", "arguments": {"question": "cached"}},
        "execution": None,
        "response": "cached answer",
    }

    async def fake_get(key):
        if key.startswith("llm:resp:") and not key.endswith(":idx"):
            return json.dumps(cached_payload).encode("utf-8")
        return None

    fake_redis.get = AsyncMock(side_effect=fake_get)
    fake_redis.rpush = AsyncMock()
    fake_redis.expire = AsyncMock()
    fake_redis.set = AsyncMock()
    fake_redis.setex = AsyncMock()
    fake_redis.publish = AsyncMock()

    fake_resolve = AsyncMock()
    monkeypatch.setattr("app.chat.service.resolve_intent", fake_resolve)

    task_id = uuid.uuid4()
    user_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    team_id = uuid.uuid4()

    await service.run_local_chat_task(fake_redis, task_id, user_id, conv_id, "查餐饮", team_id)

    fake_resolve.assert_not_awaited()
