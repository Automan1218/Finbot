import json
import uuid
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import select

from app.ab_test.bucket import assign_prompt_version
from app.agent.context import load_context_parallel
from app.agent.executor import execute_intent
from app.agent.llm import resolve_intent
from app.agent.prompt import build_prompt
from app.agent.streaming import stream_llm_to_sse
from app.agent.tools import AgentIntent
from app.cache.budget import current_year_month
from app.cache.llm_response import get_cached_response, set_cached_response
from app.cache.session import build_session_window
from app.core.database import get_db_session
from app.models.prompt_version import PromptVersion

SESSION_TTL_SECONDS = 24 * 60 * 60
TASK_TTL_SECONDS = 60 * 60
DEFAULT_PROMPT_VERSION = "v1.0"


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


async def get_compressed_window(
    redis: Redis,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
) -> list[dict[str, Any]]:
    return await build_session_window(redis, user_id, conversation_id)


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


async def _select_prompt_version(redis: Redis, team_id: uuid.UUID | None) -> str:
    if team_id is None:
        return DEFAULT_PROMPT_VERSION
    if not hasattr(redis, "get"):
        return DEFAULT_PROMPT_VERSION
    return await assign_prompt_version(str(team_id), redis, default=DEFAULT_PROMPT_VERSION)


async def _load_prompt_system_text(prompt_version: str) -> str | None:
    if prompt_version == DEFAULT_PROMPT_VERSION:
        return None
    async with get_db_session() as db:
        result = await db.execute(
            select(PromptVersion).where(PromptVersion.version == prompt_version)
        )
        row = result.scalar_one_or_none()
        if row and row.is_active:
            return row.system_txt
    return None


async def run_local_chat_task(
    redis: Redis,
    task_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    team_id: uuid.UUID | None = None,
) -> None:
    try:
        year_month = current_year_month()
        prompt_version = await _select_prompt_version(redis, team_id)
        if team_id is not None:
            cached = await get_cached_response(redis, message, str(team_id), year_month)
            if cached is not None:
                await append_message(redis, user_id, conversation_id, "assistant", cached["response"])
                await push_task_event(
                    redis,
                    task_id,
                    conversation_id,
                    step="complete",
                    status="done",
                    message=cached["response"],
                    data={
                        "intent": cached.get("intent"),
                        "execution": cached.get("execution"),
                        "cache": "hit",
                        "prompt_version": cached.get("prompt_version", prompt_version),
                    },
                )
                return

        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="agent",
            status="running",
            message="Parsing request",
        )
        system_prompt = await _load_prompt_system_text(prompt_version)
        intent, intent_source = await resolve_intent(message, system_prompt=system_prompt)
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
            data={
                "intent": intent,
                "intent_source": intent_source,
                "execution": execution,
                "prompt_version": prompt_version,
            },
        )
        if team_id is not None and intent.get("name") != "clarify":
            await set_cached_response(
                redis,
                message,
                str(team_id),
                year_month,
                {
                    "intent": intent,
                    "execution": execution,
                    "response": response,
                    "prompt_version": prompt_version,
                },
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


async def run_streaming_chat_task(
    redis: Redis,
    task_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    message: str,
    team_id: uuid.UUID | None = None,
) -> None:
    try:
        prompt_version = await _select_prompt_version(redis, team_id)
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="load_context",
            status="running",
            message="Loading context",
        )
        async with get_db_session() as db:
            ctx = await load_context_parallel(
                redis=redis,
                db=db,
                user_id=user_id,
                conversation_id=conversation_id,
                team_id=team_id,
                query=message,
            )

        system_prompt = await _load_prompt_system_text(prompt_version)
        rag_text = "\n".join(
            str(chunk.get("chunk_text", ""))
            for chunk in ctx.get("rag", [])
            if chunk.get("chunk_text")
        )
        prompt = build_prompt(
            history=ctx.get("history", []),
            rag_context=rag_text,
            user_msg=message,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
        )

        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="generating",
            status="running",
            message="Generating",
            data={"prompt_version": prompt_version},
        )
        full_text = await stream_llm_to_sse(
            redis=redis,
            task_id=task_id,
            conversation_id=conversation_id,
            messages=prompt,
        )

        await append_message(redis, user_id, conversation_id, "assistant", full_text)
        await push_task_event(
            redis,
            task_id,
            conversation_id,
            step="complete",
            status="done",
            message=full_text,
            data={"mode": "streaming", "prompt_version": prompt_version},
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
    if intent["name"] == "record_batch":
        items = intent["arguments"].get("transactions") or []
        return f"已解析 {len(items)} 笔批量记账。"
    if intent["name"] == "query_transactions":
        args = intent["arguments"]
        return f"已解析为查询：{args['date_from']} 至 {args['date_to']}。"
    if intent["name"] == "analyze_spending":
        args = intent["arguments"]
        return (
            f"已解析为支出分析：{args['period_start']} 至 {args['period_end']}，"
            f"按 {args['group_by']} 汇总。"
        )
    if intent["name"] == "check_budget":
        args = intent["arguments"]
        scope = args.get("category") or "全部分类"
        return f"已解析为预算检查：{scope}。"
    if intent["name"] == "generate_report":
        args = intent["arguments"]
        return f"已解析为报表请求：{args['period_start']} 至 {args['period_end']}，按 {args['group_by']} 汇总。"
    return str(intent["arguments"]["question"])
