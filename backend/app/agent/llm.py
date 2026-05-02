import json
from datetime import date
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


async def resolve_intent(message: str) -> tuple[AgentIntent, str]:
    try:
        return await detect_intent_with_openai(message), "openai"
    except OpenAIIntentUnavailable:
        return detect_intent(message), "rules"
    except Exception:
        return detect_intent(message), "rules"


async def detect_intent_with_openai(
    message: str,
    client: AsyncOpenAI | None = None,
    model: str | None = None,
) -> AgentIntent:
    if client is None and not settings.OPENAI_API_KEY:
        raise OpenAIIntentUnavailable("OPENAI_API_KEY is not configured")

    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model or settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    if name == "record_transaction":
        return _normalize_record_transaction(arguments, original_message)
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
