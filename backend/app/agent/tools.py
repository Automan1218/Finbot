from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import re
from typing import Any, Literal, TypedDict


class AgentIntent(TypedDict):
    name: Literal["record_transaction", "generate_report", "rag_retrieve", "clarify"]
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

FINBOT_TOOLS = [RECORD_TRANSACTION_TOOL, GENERATE_REPORT_TOOL, RAG_RETRIEVE_TOOL, CLARIFY_TOOL]

_AMOUNT_RE = re.compile(r"(?:￥|¥|RMB|人民币)?\s*(\d+(?:\.\d{1,2})?)\s*(?:元|块|yuan)?", re.I)
_EXPENSE_WORDS = ("花", "买", "支出", "消费", "付", "付款", "扣款")
_INCOME_WORDS = ("收入", "工资", "收到", "进账", "报销")
_REPORT_WORDS = ("报表", "报告", "总结", "统计")
_RAG_WORDS = ("知识库", "政策", "制度", "标准", "流程", "限制", "规定")


def detect_intent(message: str) -> AgentIntent:
    text = message.strip()
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
