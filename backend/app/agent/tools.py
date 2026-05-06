from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Any, Literal, TypedDict


class AgentIntent(TypedDict):
    name: Literal[
        "record_transaction",
        "record_batch",
        "query_transactions",
        "analyze_spending",
        "check_budget",
        "generate_report",
        "rag_retrieve",
        "clarify",
    ]
    arguments: dict[str, Any]


RECORD_TRANSACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_transaction",
        "description": "Create a structured finance transaction from natural language.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount_yuan": {"type": "number"},
                "direction": {"type": "string", "enum": ["income", "expense"]},
                "category": {"type": "string"},
                "account_name": {"type": "string"},
                "transaction_date": {"type": "string", "format": "date"},
                "description": {"type": "string"},
            },
            "required": [
                "amount_yuan",
                "direction",
                "category",
                "account_name",
                "transaction_date",
                "description",
            ],
        },
    },
}

RECORD_BATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "record_batch",
        "description": (
            "Create multiple finance transactions in a single call when the user "
            "describes several money movements in one message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "transactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "amount_yuan": {"type": "number"},
                            "direction": {"type": "string", "enum": ["income", "expense"]},
                            "category": {"type": "string"},
                            "account_name": {"type": "string"},
                            "transaction_date": {"type": "string", "format": "date"},
                            "description": {"type": "string"},
                        },
                        "required": [
                            "amount_yuan",
                            "direction",
                            "category",
                            "account_name",
                            "transaction_date",
                            "description",
                        ],
                    },
                }
            },
            "required": ["transactions"],
        },
    },
}

QUERY_TRANSACTIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "query_transactions",
        "description": "List historical transactions filtered by period, category, account, or direction.",
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "format": "date"},
                "date_to": {"type": "string", "format": "date"},
                "category": {"type": "string"},
                "account_name": {"type": "string"},
                "direction": {"type": "string", "enum": ["income", "expense"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": [],
        },
    },
}

ANALYZE_SPENDING_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_spending",
        "description": (
            "Analyze spending trends across a period and optionally compare to "
            "the previous equal-length period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["last_7_days", "last_30_days", "this_month", "custom"],
                },
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "group_by": {
                    "type": "string",
                    "enum": ["category", "account", "day", "week"],
                },
                "compare_with": {"type": "string", "enum": ["prev_period"]},
            },
            "required": ["period"],
        },
    },
}

CHECK_BUDGET_TOOL = {
    "type": "function",
    "function": {
        "name": "check_budget",
        "description": "Report current budget usage, optionally filtered by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
            },
            "required": [],
        },
    },
}

GENERATE_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": "Generate a finance report for a period.",
        "parameters": {
            "type": "object",
            "properties": {
                "period_start": {"type": "string", "format": "date"},
                "period_end": {"type": "string", "format": "date"},
                "group_by": {"type": "string", "enum": ["category", "account", "day"]},
            },
            "required": ["period_start", "period_end", "group_by"],
        },
    },
}

CLARIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "clarify",
        "description": "Ask a follow-up question when required information is missing.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "missing_fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "missing_fields"],
        },
    },
}

RAG_RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "rag_retrieve",
        "description": "Look up team policy or knowledge documents to answer a question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}

FINBOT_TOOLS = [
    RECORD_TRANSACTION_TOOL,
    RECORD_BATCH_TOOL,
    QUERY_TRANSACTIONS_TOOL,
    ANALYZE_SPENDING_TOOL,
    CHECK_BUDGET_TOOL,
    GENERATE_REPORT_TOOL,
    RAG_RETRIEVE_TOOL,
    CLARIFY_TOOL,
]

_AMOUNT_RE = re.compile(r"(?:￥|¥|RMB|人民币)?\s*(\d+(?:\.\d{1,2})?)\s*(?:元|块|yuan)?", re.I)
_ANY_AMOUNT_RE = re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:元|块|yuan|rmb|cny)?", re.I)
_EXPENSE_WORDS = ("花", "买", "支出", "消费", "付", "付款", "扣款")
_INCOME_WORDS = ("收入", "工资", "收到", "进账", "报销")
_REPORT_WORDS = ("报表", "报告", "总结", "统计", "report", "summary")
_RAG_WORDS = ("知识库", "政策", "制度", "标准", "流程", "限制", "规定")
_QUERY_WORDS = ("查", "查询", "看看", "show", "find", "list", "history", "transaction")
_ANALYZE_WORDS = ("分析", "趋势", "对比", "相比", "compare", "analyze", "trend", "spend", "spending")
_BUDGET_WORDS = ("预算", "还剩", "剩余", "budget", "remaining", "left")


def detect_intent(message: str) -> AgentIntent:
    text = message.strip()
    lower_text = text.lower()
    batch_items = _extract_batch_items(text)
    if len(batch_items) >= 2:
        return {"name": "record_batch", "arguments": {"transactions": batch_items}}

    if any(word in lower_text or word in text for word in _BUDGET_WORDS):
        return {"name": "check_budget", "arguments": {"category": _infer_category_hint(text)}}

    if any(word in lower_text or word in text for word in _ANALYZE_WORDS):
        today = date.today()
        period = "last_7_days" if any(word in lower_text for word in ("7", "week")) else "last_30_days"
        period_start = (today - timedelta(days=6)).isoformat()
        if "月" in text or "month" in lower_text:
            period = "this_month"
            period_start = today.replace(day=1).isoformat()
        elif period == "last_30_days":
            period_start = (today - timedelta(days=29)).isoformat()
        return {
            "name": "analyze_spending",
            "arguments": {
                "period": period,
                "period_start": period_start,
                "period_end": today.isoformat(),
                "group_by": "day" if any(word in lower_text or word in text for word in ("daily", "day", "每天")) else "category",
                "compare_with": "prev_period" if any(word in lower_text or word in text for word in ("compare", "对比", "相比", "上")) else None,
            },
        }

    if any(word in lower_text or word in text for word in _QUERY_WORDS):
        return {
            "name": "query_transactions",
            "arguments": {
                "date_from": date.today().replace(day=1).isoformat(),
                "date_to": date.today().isoformat(),
                "direction": (
                    "income"
                    if "income" in lower_text or "收入" in text
                    else "expense"
                    if any(word in lower_text or word in text for word in ("expense", "spending", "支出", "花"))
                    else None
                ),
                "category": _infer_category_hint(text),
                "account_name": infer_account_name(text),
                "limit": 20,
            },
        }

    if any(word in text for word in _REPORT_WORDS):
        return {
            "name": "generate_report",
            "arguments": {
                "period_start": date.today().replace(day=1).isoformat(),
                "period_end": date.today().isoformat(),
                "group_by": "category",
            },
        }

    if any(word in text for word in _RAG_WORDS):
        return {
            "name": "rag_retrieve",
            "arguments": {"query": text},
        }

    if any(word in text for word in (*_EXPENSE_WORDS, *_INCOME_WORDS)):
        amount = _extract_amount_yuan(text)
        if amount is None:
            return {
                "name": "clarify",
                "arguments": {
                    "question": "这笔记录的金额是多少？",
                    "missing_fields": ["amount_yuan"],
                },
            }
        direction = "income" if any(word in text for word in _INCOME_WORDS) else "expense"
        return {
            "name": "record_transaction",
            "arguments": {
                "amount_yuan": float(amount),
                "amount_fen": yuan_to_fen(amount),
                "direction": direction,
                "category": infer_category(text, direction),
                "account_name": infer_account_name(text),
                "transaction_date": date.today().isoformat(),
                "description": text,
            },
        }

    return {
        "name": "clarify",
        "arguments": {
            "question": "你想记账、生成报表，还是查询预算？",
            "missing_fields": ["intent"],
        },
    }


def yuan_to_fen(amount_yuan: Decimal) -> int:
    fen = (amount_yuan * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(fen)


def infer_category(text: str, direction: str) -> str:
    if direction == "income":
        if "工资" in text:
            return "工资"
        if "报销" in text:
            return "报销"
        return "收入"
    if any(word in text for word in ("饭", "餐", "咖啡", "奶茶", "外卖")):
        return "餐饮"
    if any(word in text for word in ("地铁", "打车", "公交", "油费")):
        return "交通"
    if any(word in text for word in ("电影", "游戏", "会员")):
        return "娱乐"
    return "未分类"


def infer_account_name(text: str) -> str:
    if "支付宝" in text:
        return "支付宝"
    if "微信" in text:
        return "微信"
    if "现金" in text:
        return "现金"
    if any(word in text for word in ("银行卡", "信用卡")):
        return "银行卡"
    return "默认账户"


def _extract_amount_yuan(text: str) -> Decimal | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    return Decimal(match.group(1))


def _extract_batch_items(text: str) -> list[dict[str, Any]]:
    matches = list(_ANY_AMOUNT_RE.finditer(text))
    if len(matches) < 2:
        return []
    today = date.today().isoformat()
    account_name = infer_account_name(text)
    items: list[dict[str, Any]] = []
    for match in matches:
        amount = Decimal(match.group(1))
        description = _description_around_amount(text, match.start(), match.end())
        items.append(
            {
                "amount_yuan": float(amount),
                "amount_fen": yuan_to_fen(amount),
                "direction": "expense",
                "category": _infer_category_hint(description) or infer_category(description, "expense"),
                "account_name": account_name,
                "transaction_date": today,
                "description": description or text,
            }
        )
    return items


def _description_around_amount(text: str, start: int, end: int) -> str:
    left = max(0, start - 12)
    right = min(len(text), end + 6)
    return text[left:right].strip(" ,.;，。；")


def _infer_category_hint(text: str) -> str | None:
    lower_text = text.lower()
    if any(word in lower_text or word in text for word in ("lunch", "dinner", "coffee", "meal", "food", "饭", "餐", "咖啡", "奶茶", "外卖")):
        return "Food & Beverage"
    if any(word in lower_text or word in text for word in ("taxi", "subway", "train", "flight", "parking", "打车", "地铁", "公交", "油费")):
        return "Transportation"
    if any(word in lower_text or word in text for word in ("hotel", "lodging", "酒店", "住宿")):
        return "Lodging"
    if any(word in lower_text for word in ("software", "saas", "hosting", "cloud")):
        return "Software"
    return None
