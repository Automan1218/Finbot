import json
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings

SESSION_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_HISTORY = 10
DEFAULT_COMPRESS_AT = 20

_SUMMARIZE_PROMPT = (
    "概括以下对话历史的关键信息，保留用户意图、已确认事实、未解决问题。"
    "限制在 200 字以内，使用第三人称描述：\n\n{transcript}"
)


def _session_key(user_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
    return f"session:{user_id}:{conversation_id}"


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


async def summarize_messages(
    messages: list[dict[str, Any]],
    client: AsyncOpenAI | None = None,
) -> str:
    if not messages:
        return ""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "user", "content": _SUMMARIZE_PROMPT.format(transcript=transcript)}
        ],
    )
    return (response.choices[0].message.content or "").strip()


async def build_session_window(
    redis: Any,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    max_history: int = DEFAULT_MAX_HISTORY,
    compress_at: int = DEFAULT_COMPRESS_AT,
    client: AsyncOpenAI | None = None,
) -> list[dict[str, Any]]:
    key = _session_key(user_id, conversation_id)
    raw_messages = await redis.lrange(key, 0, -1)
    messages = [json.loads(_decode(item)) for item in raw_messages]

    if len(messages) <= compress_at:
        return messages[-max_history:] if len(messages) > max_history else messages

    older = messages[:-max_history]
    summary_text = await summarize_messages(older, client=client)
    summary_msg = {"role": "system", "content": f"历史摘要: {summary_text}"}
    recent = messages[-max_history:]

    await redis.delete(key)
    for msg in [summary_msg, *recent]:
        await redis.rpush(key, json.dumps(msg, ensure_ascii=False))
    await redis.expire(key, SESSION_TTL_SECONDS)
    return [summary_msg, *recent]
