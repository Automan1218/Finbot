import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from app.core.config import settings

TASK_TTL_SECONDS = 60 * 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_events_key(task_id: uuid.UUID) -> str:
    return f"chat-task-events:{task_id}"


def _task_key(task_id: uuid.UUID) -> str:
    return f"chat-task:{task_id}"


async def _publish_event(redis: Any, task_id: uuid.UUID, event: dict[str, Any]) -> None:
    payload = json.dumps(event, ensure_ascii=False)
    await redis.publish(f"task-progress:{task_id}", payload)
    if hasattr(redis, "rpush"):
        await redis.rpush(_task_events_key(task_id), payload)
    if hasattr(redis, "expire"):
        await redis.expire(_task_events_key(task_id), TASK_TTL_SECONDS)
    if hasattr(redis, "set"):
        await redis.set(_task_key(task_id), payload, ex=TASK_TTL_SECONDS)


async def _iter_openai_stream(
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    model: str,
) -> AsyncIterator[Any]:
    stream_method = getattr(client.chat.completions, "stream", None)
    if callable(stream_method):
        async with stream_method(model=model, messages=messages, stream=True) as stream:
            async for chunk in stream:
                yield chunk
        return

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        yield chunk


async def stream_llm_to_sse(
    redis: Any,
    task_id: uuid.UUID,
    conversation_id: uuid.UUID,
    messages: list[dict[str, Any]],
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> str:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    chunks: list[str] = []

    async for chunk in _iter_openai_stream(
        client=client,
        messages=messages,
        model=model or settings.OPENAI_MODEL,
    ):
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if not content:
            continue
        chunks.append(content)
        await _publish_event(
            redis,
            task_id,
            {
                "task_id": str(task_id),
                "conversation_id": str(conversation_id),
                "step": "generating",
                "status": "running",
                "message": content,
                "data": {"type": "delta"},
                "delta": content,
                "created_at": _now_iso(),
            },
        )

    return "".join(chunks)
