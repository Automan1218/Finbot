from openai import AsyncOpenAI

from app.core.config import settings

FAILURE_TYPES: tuple[str, ...] = (
    "retrieval_failure",
    "intent_failure",
    "generation_failure",
)

_PROMPT = (
    "Classify the failure of an AI finance assistant reply into exactly one label.\n"
    "Labels:\n"
    "- retrieval_failure: knowledge retrieval returned wrong or missing documents.\n"
    "- intent_failure: intent recognition was wrong, fields missing, or category wrong.\n"
    "- generation_failure: retrieval was correct but the generated answer is poor.\n"
    "Return only the label string.\n\n"
    "User question: {query}\n"
    "AI reply: {response}\n"
    "User correction: {correction}\n"
    "Label:"
)


def fallback_classify(query: str, response: str, correction: str | None) -> str:
    if correction:
        return "retrieval_failure"
    return "generation_failure"


async def classify_failure(
    query: str,
    response: str,
    correction: str | None,
    client: AsyncOpenAI | None = None,
) -> str:
    client = client or AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    try:
        resp = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(
                        query=query,
                        response=response,
                        correction=correction or "(none)",
                    ),
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        return fallback_classify(query, response, correction)

    for label in FAILURE_TYPES:
        if label in raw:
            return label
    return fallback_classify(query, response, correction)
