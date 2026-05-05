import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.feedback.classifier import classify_failure
from app.models.feedback import Feedback


def _dict_values_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    return " ".join(str(value) for value in payload.values())


async def record_feedback(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    task_type: str | None,
    rating: int,
    feedback_type: str,
    original_output: dict[str, Any] | None,
    corrected_output: dict[str, Any] | None,
    prompt_version: str | None,
    db: AsyncSession,
) -> Feedback:
    failure_type: str | None = None
    if rating < 0:
        failure_type = await classify_failure(
            query=task_type or feedback_type,
            response=_dict_values_text(original_output),
            correction=_dict_values_text(corrected_output) or None,
        )

    row = Feedback(
        team_id=team_id,
        user_id=user_id,
        task_id=task_id,
        task_type=task_type,
        rating=rating,
        feedback_type=feedback_type,
        failure_type=failure_type,
        original_output=original_output,
        corrected_output=corrected_output,
        prompt_version=prompt_version,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def record_implicit_signal(
    team_id: uuid.UUID,
    user_id: uuid.UUID,
    task_id: uuid.UUID | None,
    task_type: str | None,
    signal_type: str,
    prompt_version: str | None,
    db: AsyncSession,
) -> Feedback:
    rating = 1 if signal_type == "accepted" else -1
    row = Feedback(
        team_id=team_id,
        user_id=user_id,
        task_id=task_id,
        task_type=task_type,
        rating=rating,
        feedback_type=signal_type,
        prompt_version=prompt_version,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def list_feedback(team_id: uuid.UUID, db: AsyncSession) -> list[Feedback]:
    result = await db.execute(
        select(Feedback)
        .where(Feedback.team_id == team_id)
        .order_by(Feedback.created_at.desc())
    )
    return list(result.scalars().all())
