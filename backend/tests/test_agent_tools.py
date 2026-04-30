from decimal import Decimal

from app.agent.tools import detect_intent, yuan_to_fen


def test_detect_record_transaction_intent():
    intent = detect_intent("今天午饭花了 35.5 元，用支付宝")

    assert intent["name"] == "record_transaction"
    assert intent["arguments"]["amount_fen"] == 3550
    assert intent["arguments"]["direction"] == "expense"
    assert intent["arguments"]["category"] == "餐饮"
    assert intent["arguments"]["account_name"] == "支付宝"


def test_detect_report_intent():
    intent = detect_intent("生成本月报表")

    assert intent["name"] == "generate_report"
    assert intent["arguments"]["group_by"] == "category"


def test_detect_clarify_intent_when_amount_missing():
    intent = detect_intent("今天买咖啡")

    assert intent["name"] == "clarify"
    assert intent["arguments"]["missing_fields"] == ["amount_yuan"]


def test_yuan_to_fen_rounds_half_up():
    assert yuan_to_fen(Decimal("12.345")) == 1235
