import json
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from openai import AsyncOpenAI

from app.agent.tools import AgentIntent, FINBOT_TOOLS, detect_intent, yuan_to_fen
from app.core.config import settings


class OpenAIIntentUnavailable(Exception):
    pass


SYSTEM_PROMPT = """
You are Finbot's intent parser for a finance assistant.
Choose exactly one function tool.
Amounts in tool arguments use yuan. Dates must be ISO 8601 dates.
Use clarify when the user intent or required fields are missing.
Use rag_retrieve for policy, procedure, reimbursement standard, or team knowledge questions.
Keep descriptions faithful to the user's original text.
""".strip()


async def resolve_intent(
    message: str,
    system_prompt: str | None = None,
) -> tuple[AgentIntent, str]:
    try:
        return await detect_intent_with_openai(message, system_prompt=system_prompt), "openai"
    except OpenAIIntentUnavailable:
        return detect_intent(message), "rules"
    except Exception:
        return detect_intent(message), "rules"


async def detect_intent_with_openai(
    message: str,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
) -> AgentIntent:
    if client is None and not settings.OPENAI_API_KEY:
        raise OpenAIIntentUnavailable("OPENAI_API_KEY is not configured")

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model or settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        tools=FINBOT_TOOLS,
        tool_choice="required",
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise OpenAIIntentUnavailable("OpenAI returned no choices")

    tool_calls = getattr(choices[0].message, "tool_calls", None) or []
    if not tool_calls:
        raise OpenAIIntentUnavailable("OpenAI returned no tool call")

    tool_call = tool_calls[0]
    if getattr(tool_call, "type", None) != "function":
        raise OpenAIIntentUnavailable("OpenAI returned a non-function tool call")

    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise OpenAIIntentUnavailable("OpenAI returned invalid tool arguments") from exc
    return normalize_intent(name, arguments, message)


def normalize_intent(name: str, arguments: dict[str, Any], original_message: str) -> AgentIntent:
    if name == "record_batch":
        return _normalize_record_batch(arguments, original_message)
    if name == "record_transaction":
        return _normalize_record_transaction(arguments, original_message)
    if name == "query_transactions":
        return _normalize_query_transactions(arguments)
    if name == "analyze_spending":
        return _normalize_analyze_spending(arguments)
    if name == "check_budget":
        return _normalize_check_budget(arguments)
    if name == "generate_report":
        return _normalize_generate_report(arguments)
    if name == "rag_retrieve":
        return {
            "name": "rag_retrieve",
            "arguments": {"query": str(arguments.get("query") or original_message)},
        }
    if name == "clarify":
        return {
            "name": "clarify",
            "arguments": {
                "question": str(arguments.get("question") or "Please provide more details."),
                "missing_fields": list(arguments.get("missing_fields") or []),
            },
        }
    return {
        "name": "clarify",
        "arguments": {
            "question": "Please clarify whether you want to record a transaction or generate a report.",
            "missing_fields": ["intent"],
        },
    }


def _normalize_record_transaction(
    arguments: dict[str, Any], original_message: str
) -> AgentIntent:
    amount_yuan = _decimal_from_value(arguments.get("amount_yuan"))
    if amount_yuan is None:
        return {
            "name": "clarify",
            "arguments": {
                "question": "What is the transaction amount?",
                "missing_fields": ["amount_yuan"],
            },
        }

    direction = arguments.get("direction")
    if direction not in {"income", "expense"}:
        return {
            "name": "clarify",
            "arguments": {
                "question": "Is this income or an expense?",
                "missing_fields": ["direction"],
            },
        }

    transaction_date = str(arguments.get("transaction_date") or date.today().isoformat())
    return {
        "name": "record_transaction",
        "arguments": {
            "amount_yuan": float(amount_yuan),
            "amount_fen": yuan_to_fen(amount_yuan),
            "direction": direction,
            "category": str(arguments.get("category") or "Uncategorized"),
            "account_name": str(arguments.get("account_name") or "Default"),
            "transaction_date": transaction_date,
            "description": str(arguments.get("description") or original_message),
        },
    }


def _normalize_record_batch(
    arguments: dict[str, Any], original_message: str
) -> AgentIntent:
    raw_items = arguments.get("transactions") or []
    if not isinstance(raw_items, list):
        raw_items = []
    items: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        amount_yuan = _decimal_from_value(raw.get("amount_yuan"))
        if amount_yuan is None:
            continue
        direction = raw.get("direction")
        if direction not in {"income", "expense"}:
            continue
        items.append(
            {
                "amount_yuan": float(amount_yuan),
                "amount_fen": yuan_to_fen(amount_yuan),
                "direction": direction,
                "category": str(raw.get("category") or "Uncategorized"),
                "account_name": str(raw.get("account_name") or "Default"),
                "transaction_date": str(
                    raw.get("transaction_date") or date.today().isoformat()
                ),
                "description": str(raw.get("description") or original_message),
            }
        )
    if not items:
        return {
            "name": "clarify",
            "arguments": {
                "question": "Please describe at least one transaction with amount and direction.",
                "missing_fields": ["transactions"],
            },
        }
    return {"name": "record_batch", "arguments": {"transactions": items}}


def _normalize_query_transactions(arguments: dict[str, Any]) -> AgentIntent:
    today = date.today()
    direction = arguments.get("direction")
    if direction not in {"income", "expense"}:
        direction = None
    raw_limit = arguments.get("limit")
    try:
        limit = int(raw_limit) if raw_limit is not None else 20
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 100))
    return {
        "name": "query_transactions",
        "arguments": {
            "date_from": str(arguments.get("date_from") or today.replace(day=1).isoformat()),
            "date_to": str(arguments.get("date_to") or today.isoformat()),
            "direction": direction,
            "category": arguments.get("category") or None,
            "account_name": arguments.get("account_name") or None,
            "limit": limit,
        },
    }


def _normalize_analyze_spending(arguments: dict[str, Any]) -> AgentIntent:
    today = date.today()
    period = arguments.get("period") or "last_30_days"
    if period not in {"last_7_days", "last_30_days", "this_month", "custom"}:
        period = "last_30_days"
    if period == "custom":
        period_start = str(arguments.get("period_start") or today.replace(day=1).isoformat())
        period_end = str(arguments.get("period_end") or today.isoformat())
    elif period == "last_7_days":
        period_start = (today - timedelta(days=6)).isoformat()
        period_end = today.isoformat()
    elif period == "this_month":
        period_start = today.replace(day=1).isoformat()
        period_end = today.isoformat()
    else:
        period_start = (today - timedelta(days=29)).isoformat()
        period_end = today.isoformat()

    group_by = arguments.get("group_by") or "category"
    if group_by not in {"category", "account", "day", "week"}:
        group_by = "category"

    compare_with = arguments.get("compare_with")
    if compare_with not in {"prev_period"}:
        compare_with = None

    return {
        "name": "analyze_spending",
        "arguments": {
            "period": period,
            "period_start": period_start,
            "period_end": period_end,
            "group_by": group_by,
            "compare_with": compare_with,
        },
    }


def _normalize_check_budget(arguments: dict[str, Any]) -> AgentIntent:
    raw_category = arguments.get("category")
    category = str(raw_category).strip() if raw_category else None
    return {
        "name": "check_budget",
        "arguments": {"category": category or None},
    }


def _normalize_generate_report(arguments: dict[str, Any]) -> AgentIntent:
    today = date.today()
    group_by = arguments.get("group_by")
    if group_by not in {"category", "account", "day"}:
        group_by = "category"
    return {
        "name": "generate_report",
        "arguments": {
            "period_start": str(arguments.get("period_start") or today.replace(day=1).isoformat()),
            "period_end": str(arguments.get("period_end") or today.isoformat()),
            "group_by": group_by,
        },
    }


def _decimal_from_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
