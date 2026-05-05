import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app


@pytest.fixture
async def client(db_session, finance_setup):
    user, team, token = finance_setup

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as ac:
        yield ac, team
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_post_feedback_records_row(client, monkeypatch):
    ac, team = client
    fake_classify = AsyncMock(return_value="intent_failure")
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    response = await ac.post(
        f"/feedback?team_id={team.id}",
        json={
            "team_id": str(team.id),
            "task_id": str(uuid.uuid4()),
            "task_type": "record_transaction",
            "rating": -1,
            "feedback_type": "correction",
            "original_output": {"category": "Other"},
            "corrected_output": {"category": "Food"},
            "prompt_version": "v1.0",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["rating"] == -1
    assert body["failure_type"] == "intent_failure"


@pytest.mark.asyncio
async def test_post_feedback_skips_classifier_for_positive(client, monkeypatch):
    ac, team = client
    fake_classify = AsyncMock()
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    response = await ac.post(
        f"/feedback?team_id={team.id}",
        json={
            "team_id": str(team.id),
            "rating": 1,
            "feedback_type": "thumbs",
        },
    )

    assert response.status_code == 201
    assert response.json()["failure_type"] is None
    fake_classify.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_feedback_filters_by_team(client, monkeypatch):
    ac, team = client
    fake_classify = AsyncMock(return_value="generation_failure")
    monkeypatch.setattr("app.feedback.service.classify_failure", fake_classify)

    await ac.post(
        f"/feedback?team_id={team.id}",
        json={
            "team_id": str(team.id),
            "rating": -1,
            "feedback_type": "thumbs",
        },
    )
    listed = await ac.get(f"/feedback?team_id={team.id}")

    assert listed.status_code == 200
    items = listed.json()
    assert len(items) >= 1
    assert all(item["team_id"] == str(team.id) for item in items)
