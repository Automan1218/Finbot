import json
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.agent.executor import execute_intent
from app.agent.llm import resolve_intent
from app.agent.tools import AgentIntent
from app.core.database import get_db_session

SESSION_TTL_SECONDS = 24 * 60 * 60
TASK_TTL_SECONDS = 60 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _decode(value: bytes | str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _session_key(user_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
    return f"session:{user_id}:{conversation_id}"


def _task_key(task_id: uuid.UUID) -> str:
    return f"chat-task:{task_id}"


def _task_events_key(task_id: uuid.UUID) -> str:
    return f"chat-task-events:{task_id}"


def _task_channel(task_id: uuid.UUID) -> str:
    return f"task-progress:{task_id}"


async def append_message(
    redis: Redis,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
) -> dict[str, Any]:
    message = {
        "role": role,
        "content": content,
        "created_at": _now().isoformat(),
    }
    key = _session_key(user_id, conversation_id)
    await redis.rpush(key, json.dumps(message))
    await redis.expire(key, SESSION_TTL_SECONDS)
    return message


async def get_history(
    redis: Redis,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    page: int = 1,
    size: int = 20,
) -> list[dict[str, Any]]:
    start = max(page - 1, 0) * size
    stop = start + size - 1
    raw_messages = await redis.lrange(_session_key(user_id, conversation_id), start, stop)
    return [json.loads(_decode(item)) for item in raw_messages]


async def clear_history(redis: Redis, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
    await redis.delete(_session_key(user_id, conversation_id))


async def push_task_event(
    redis: Redis,
    task_id: uuid.UUID,
    conversation_id: uuid.UUID,
    step: str,
    status: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = {
        "task_id": str(task_id),
        "conversation_id": str(conversation_id),
        "step": step,
        "status": status,
        "message": message,
        "data": data,
        "created_at": _now().isoformat(),
    }
    payload = json.dumps(event)
    await redis.rpush(_task_events_key(task_id), payload)
    await redis.expire(_task_events_key(task_id), TASK_TTL_SECONDS)
    await redis.set(_task_key(task_id), payload, ex=TASK_TTL_SECONDS)
    await redis.publish(_task_channel(task_id), payload)
    return event


async def list_task_events(redis: Redis, task_id: uuid.UUID, start: int = 0) -> list[dict[str, Any]]:
    raw_events = await redis.lrange(_task_events_key(task_id), start, -1)
    return [json.loads(_decode(item)) for item in raw_events]


async def create_chat_task(
    redis: Redis,
    user_id: uuid.UUID,
    team_id: uuid.UUID,
    message: str,
    conversation_id: uuid.UUID | None,
) -> tuple[uuid.UUID, uuid.UUID]:
    task_id = uuid.uuid4()
    conversation_id = conversation_id or uuid.uuid4()
    await append_message(redis, user_id, conversation_id, "user", message)
    await push_task_event(
        redis,
        task_id,
        conversation_id,
        step="queued",
        status="queued",
        message="Message accepted",
        data={"team_id": str(team_id)},
    )
    return task_id, conversation_id


async def run_local_chat_task(
    redis: Redis,
    task_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    team_id: uuid.UUID | None = None,
) -> None:
    try:
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="agent",
            status="running",
            message="Parsing request",
        )
        intent, intent_source = await resolve_intent(message)
        execution = None
        if team_id is not None:
            await push_task_event(
                redis,
                task_id,
                conversation_id,
                step="execute",
                status="running",
                message="Executing intent",
                data={"intent": intent, "intent_source": intent_source},
            )
            async with get_db_session() as db:
                execution = await execute_intent(intent, team_id, user_id, db)
        response = _build_agent_response(intent, execution)
        await append_message(redis, user_id, conversation_id, "assistant", response)
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="complete",
            status="done",
            message=response,
            data={"intent": intent, "intent_source": intent_source, "execution": execution},
        )
    except Exception as exc:
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="error",
            status="error",
            message=str(exc),
        )


def _build_agent_response(
    intent: AgentIntent, execution: dict[str, Any] | None = None
) -> str:
    if execution:
        return str(execution["message"])
    if intent["name"] == "record_transaction":
        args = intent["arguments"]
        return (
            f"已解析为{args['direction']}记录：{args['category']} "
            f"{args['amount_fen']} 分，账户 {args['account_name']}。"
        )
    if intent["name"] == "generate_report":
        args = intent["arguments"]
        return f"已解析为报表请求：{args['period_start']} 至 {args['period_end']}，按 {args['group_by']} 汇总。"
    return str(intent["arguments"]["question"])
