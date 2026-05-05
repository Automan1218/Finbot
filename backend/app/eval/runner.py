from collections import defaultdict
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.llm import resolve_intent
from app.eval.golden_set import EVAL_CASES
from app.eval.metrics import score_scenario


async def run_eval_cases(
    cases: list[dict[str, Any]],
    runner: Callable[[str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    scenario_scores: dict[str, list[float]] = defaultdict(list)
    errors = 0

    for case in cases:
        scenario = case["scenario"]
        try:
            actual = await runner(case["input"])
        except Exception:
            scenario_scores[scenario].append(0.0)
            errors += 1
            continue
        score = score_scenario(scenario, actual, case["expect"])
        scenario_scores[scenario].append(score)

    scenarios: dict[str, dict[str, float]] = {}
    weighted_total = 0.0
    total_count = 0
    for scenario, scores in scenario_scores.items():
        avg = sum(scores) / len(scores)
        scenarios[scenario] = {"count": len(scores), "avg_score": avg}
        weighted_total += sum(scores)
        total_count += len(scores)

    aggregate = weighted_total / total_count if total_count else 0.0
    return {
        "scenarios": scenarios,
        "aggregate": aggregate,
        "total": total_count,
        "errors": errors,
    }


async def eval_prompt_version(
    version_id: str,
    db: AsyncSession,
    redis: Any,
) -> dict[str, Any]:
    async def runner(text: str) -> dict[str, Any]:
        intent, _ = await resolve_intent(text)
        payload: dict[str, Any] = {"fn": intent["name"]}
        payload.update(intent.get("arguments", {}))
        return payload

    result = await run_eval_cases(EVAL_CASES, runner=runner)
    result["prompt_version"] = version_id
    return result
