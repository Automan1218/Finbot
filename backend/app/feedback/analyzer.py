import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.feedback import Feedback


async def aggregate_failure_modes(
    team_id: uuid.UUID,
    since: datetime,
    db: AsyncSession,
) -> dict[str, int]:
    result = await db.execute(
        select(Feedback.failure_type, func.count())
        .where(
            Feedback.team_id == team_id,
            Feedback.rating < 0,
            Feedback.created_at >= since,
        )
        .group_by(Feedback.failure_type)
    )
    counts: dict[str, int] = {}
    total = 0
    for failure_type, count in result.all():
        if failure_type:
            counts[str(failure_type)] = int(count)
        total += int(count)
    counts["total_negative"] = total
    return counts
