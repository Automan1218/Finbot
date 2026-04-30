import json
from types import SimpleNamespace

import pytest

from app.agent.llm import detect_intent_with_openai, normalize_intent, resolve_intent


class FakeCompletions:
    def __init__(self, tool_name, arguments):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        tool_call = SimpleNamespace(
            type="function",
            function=SimpleNamespace(
                name=self.tool_name,
                arguments=json.dumps(self.arguments),
            ),
        )
        message = SimpleNamespace(tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, tool_name, arguments):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(tool_name, arguments)
        )


@pytest.mark.asyncio
async def test_detect_intent_with_openai_parses_tool_call():
    client = FakeClient(
        "record_transaction",
        {
            "amount_yuan": 35.5,
            "direction": "expense",
            "category": "餐饮",
            "account_name": "支付宝",
            "transaction_date": "2026-05-01",
            "description": "午饭",
        },
    )

    intent = await detect_intent_with_openai(
        "今天午饭花了 35.5 元", client=client, model="test-model"
    )

    assert intent["name"] == "record_transaction"
    assert intent["arguments"]["amount_fen"] == 3550
    assert client.chat.completions.calls[0]["tools"]
    assert client.chat.completions.calls[0]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_resolve_intent_falls_back_without_api_key(monkeypatch):
    monkeypatch.setattr("app.agent.llm.settings.OPENAI_API_KEY", "")

    intent, source = await resolve_intent("生成本月报表")

    assert source == "rules"
    assert intent["name"] == "generate_report"


def test_normalize_intent_clarifies_missing_amount():
    intent = normalize_intent(
        "record_transaction",
        {"direction": "expense", "category": "餐饮"},
        "今天买咖啡",
    )

    assert intent["name"] == "clarify"
    assert intent["arguments"]["missing_fields"] == ["amount_yuan"]
