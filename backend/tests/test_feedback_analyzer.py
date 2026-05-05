from datetime import datetime, timedelta, timezone

import pytest

from app.feedback.analyzer import aggregate_failure_modes
from app.models.feedback import Feedback


@pytest.mark.asyncio
async def test_aggregate_failure_modes_counts_each_label(db_session, finance_setup):
    user, team, _ = finance_setup
    rows = [
        Feedback(
            team_id=team.id,
            user_id=user.id,
            rating=-1,
            feedback_type="thumbs",
            failure_type="retrieval_failure",
        ),
        Feedback(
            team_id=team.id,
            user_id=user.id,
            rating=-1,
            feedback_type="thumbs",
            failure_type="retrieval_failure",
        ),
        Feedback(
            team_id=team.id,
            user_id=user.id,
            rating=-1,
            feedback_type="correction",
            failure_type="intent_failure",
        ),
        Feedback(
            team_id=team.id,
            user_id=user.id,
            rating=1,
            feedback_type="thumbs",
            failure_type=None,
        ),
    ]
    for row in rows:
        db_session.add(row)
    await db_session.commit()

    since = datetime.now(timezone.utc) - timedelta(days=7)
    result = await aggregate_failure_modes(team.id, since.replace(tzinfo=None), db_session)

    assert result["retrieval_failure"] == 2
    assert result["intent_failure"] == 1
    assert result.get("generation_failure", 0) == 0
    assert result["total_negative"] == 3


@pytest.mark.asyncio
async def test_aggregate_excludes_old_rows(db_session, finance_setup):
    user, team, _ = finance_setup
    old = Feedback(
        team_id=team.id,
        user_id=user.id,
        rating=-1,
        feedback_type="thumbs",
        failure_type="retrieval_failure",
    )
    db_session.add(old)
    await db_session.commit()

    since = datetime.now(timezone.utc) + timedelta(days=1)
    result = await aggregate_failure_modes(team.id, since.replace(tzinfo=None), db_session)
    assert result["total_negative"] == 0
