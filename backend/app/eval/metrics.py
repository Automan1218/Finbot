from typing import Any


def field_accuracy(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if expected.get("fn") != actual.get("fn"):
        return 0.0
    fields = list(expected.keys())
    if not fields:
        return 1.0
    matched = sum(1 for key in fields if expected.get(key) == actual.get(key))
    return matched / len(fields)


def transaction_count_match(expected_count: int, actual_count: int) -> float:
    if expected_count == 0 and actual_count == 0:
        return 1.0
    if expected_count == 0:
        return 0.0
    return min(actual_count, expected_count) / max(actual_count, expected_count)


def missing_field_recall(
    expected_fields: list[str],
    actual_fields: list[str],
) -> float:
    if not expected_fields:
        return 1.0
    expected_set = set(expected_fields)
    matched = sum(1 for field in actual_fields if field in expected_set)
    return matched / len(expected_set)


def score_scenario(
    scenario: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> float:
    if scenario == "clarify":
        if expected.get("fn") != actual.get("fn"):
            return 0.0
        expected_missing = list(expected.get("missing_fields") or [])
        actual_missing = list(actual.get("missing_fields") or [])
        if not expected_missing:
            return 1.0
        return missing_field_recall(expected_missing, actual_missing)
    if scenario == "record_batch":
        if expected.get("fn") != actual.get("fn"):
            return 0.0
        expected_count = expected.get("count")
        if expected_count is None:
            return 1.0
        return 1.0 if expected_count == actual.get("count") else 0.0
    if scenario in {"analyze_spending", "check_budget"}:
        return 1.0 if field_accuracy(expected, actual) == 1.0 else 0.0
    return field_accuracy(expected, actual)
