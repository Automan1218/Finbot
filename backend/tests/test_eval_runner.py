import pytest

from app.eval.runner import run_eval_cases


@pytest.mark.asyncio
async def test_runner_aggregates_per_scenario():
    cases = [
        {
            "id": "a",
            "scenario": "record_transaction",
            "input": "x",
            "expect": {"fn": "record_transaction", "amount_yuan": 1},
        },
        {
            "id": "b",
            "scenario": "record_transaction",
            "input": "y",
            "expect": {"fn": "record_transaction", "amount_yuan": 2},
        },
        {
            "id": "c",
            "scenario": "clarify",
            "input": "z",
            "expect": {"fn": "clarify"},
        },
    ]

    async def fake_run(text: str) -> dict:
        if text == "x":
            return {"fn": "record_transaction", "amount_yuan": 1}
        if text == "y":
            return {"fn": "record_transaction", "amount_yuan": 99}
        return {"fn": "clarify"}

    result = await run_eval_cases(cases, runner=fake_run)

    assert result["scenarios"]["record_transaction"]["count"] == 2
    assert result["scenarios"]["record_transaction"]["avg_score"] == 0.75
    assert result["scenarios"]["clarify"]["avg_score"] == 1.0
    assert 0.0 < result["aggregate"] < 1.0
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_runner_handles_runner_exception():
    async def boom(text: str):
        raise RuntimeError("oops")

    cases = [
        {
            "id": "a",
            "scenario": "record_transaction",
            "input": "x",
            "expect": {"fn": "record_transaction"},
        }
    ]
    result = await run_eval_cases(cases, runner=boom)

    assert result["scenarios"]["record_transaction"]["avg_score"] == 0.0
    assert result["errors"] == 1
