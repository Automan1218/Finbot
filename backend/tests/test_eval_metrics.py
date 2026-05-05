import pytest

from app.eval.metrics import (
    field_accuracy,
    missing_field_recall,
    score_scenario,
    transaction_count_match,
)


def test_field_accuracy_all_match():
    expected = {"fn": "record_transaction", "amount_yuan": 35, "direction": "expense"}
    actual = {"fn": "record_transaction", "amount_yuan": 35, "direction": "expense"}
    assert field_accuracy(expected, actual) == 1.0


def test_field_accuracy_partial():
    expected = {"fn": "record_transaction", "amount_yuan": 35, "category": "Food"}
    actual = {"fn": "record_transaction", "amount_yuan": 35, "category": "Other"}
    assert field_accuracy(expected, actual) == pytest.approx(2 / 3)


def test_field_accuracy_zero_when_fn_mismatch():
    expected = {"fn": "record_transaction", "amount_yuan": 35}
    actual = {"fn": "clarify"}
    assert field_accuracy(expected, actual) == 0.0


def test_transaction_count_match_exact():
    assert transaction_count_match(3, 3) == 1.0
    assert transaction_count_match(3, 2) == pytest.approx(2 / 3)
    assert transaction_count_match(0, 0) == 1.0


def test_missing_field_recall_full():
    assert missing_field_recall(["amount_yuan"], ["amount_yuan"]) == 1.0
    assert missing_field_recall(["amount_yuan", "direction"], ["amount_yuan"]) == 0.5
    assert missing_field_recall([], []) == 1.0


def test_score_scenario_dispatches():
    expected = {"fn": "record_transaction", "amount_yuan": 35}
    actual = {"fn": "record_transaction", "amount_yuan": 35}
    assert score_scenario("record_transaction", actual, expected) == 1.0

    expected_clarify = {"fn": "clarify", "missing_fields": ["amount_yuan"]}
    actual_clarify = {"fn": "clarify", "missing_fields": ["amount_yuan"]}
    assert score_scenario("clarify", actual_clarify, expected_clarify) == 1.0
