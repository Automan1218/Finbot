from unittest.mock import AsyncMock, MagicMock

import pytest

from app.feedback.classifier import (
    FAILURE_TYPES,
    classify_failure,
    fallback_classify,
)


def _build_client(label: str) -> MagicMock:
    msg = MagicMock(content=label)
    choice = MagicMock(message=msg)
    response = MagicMock(choices=[choice])
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def test_failure_types_const():
    assert FAILURE_TYPES == (
        "retrieval_failure",
        "intent_failure",
        "generation_failure",
    )


def test_fallback_classify_correction_present():
    label = fallback_classify(
        query="travel reimbursement limit",
        response="Cannot provide a specific rule",
        correction="Hotel must not exceed 500 per night",
    )
    assert label == "retrieval_failure"


def test_fallback_classify_no_correction_returns_generation():
    label = fallback_classify(
        query="Generate this month's report",
        response="unclear",
        correction=None,
    )
    assert label == "generation_failure"


@pytest.mark.asyncio
async def test_classify_failure_uses_llm_and_normalizes_label():
    client = _build_client("intent_failure\n")

    label = await classify_failure(
        query="Today lunch 35 yuan",
        response="Not recognized as expense",
        correction="Food 35 yuan expense",
        client=client,
    )

    assert label == "intent_failure"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_classify_failure_falls_back_when_label_invalid():
    client = _build_client("not-a-known-label")

    label = await classify_failure(
        query="x",
        response="y",
        correction="z",
        client=client,
    )

    assert label in FAILURE_TYPES
