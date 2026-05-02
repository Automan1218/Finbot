import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.core.security import create_access_token, hash_password
from app.main import app
from app.models.team import TeamMember
from app.models.user import User


@pytest.fixture
async def knowledge_client(db_session, finance_setup):
    user, team, token = finance_setup

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=headers,
    ) as client:
        yield client, user, team
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_document_returns_201(knowledge_client, monkeypatch):
    client, _, team = knowledge_client
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    response = await client.post(
        "/knowledge-base/documents",
        params={"team_id": str(team.id)},
        json={
            "team_id": str(team.id),
            "title": "Reimbursement Policy",
            "source_type": "policy",
            "text": "出差报销限额：机票经济舱，酒店每日500元以内。" * 10,
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Reimbursement Policy"
    assert data["source_type"] == "policy"


@pytest.mark.asyncio
async def test_list_documents_returns_team_docs(knowledge_client, monkeypatch):
    client, _, team = knowledge_client
    fake_embed = AsyncMock(return_value=[0.1] * 1536)
    monkeypatch.setattr("app.rag.ingest.get_embedding", fake_embed)

    await client.post(
        "/knowledge-base/documents",
        params={"team_id": str(team.id)},
        json={
            "team_id": str(team.id),
            "title": "Doc A",
            "source_type": "manual",
            "text": "policy content " * 50,
        },
    )
    response = await client.get(
        "/knowledge-base/documents",
        params={"team_id": str(team.id)},
    )

    assert response.status_code == 200
    docs = response.json()
    assert any(doc["title"] == "Doc A" for doc in docs)


@pytest.mark.asyncio
async def test_create_document_requires_admin(knowledge_client, db_session):
    client, _, team = knowledge_client
    member = User(
        email=f"viewer_{uuid.uuid4().hex[:6]}@test.com",
        password_hash=hash_password("pw"),
    )
    db_session.add(member)
    await db_session.flush()
    db_session.add(TeamMember(team_id=team.id, user_id=member.id, role="viewer"))
    await db_session.commit()
    viewer_token = create_access_token(str(member.id))

    response = await client.post(
        "/knowledge-base/documents",
        params={"team_id": str(team.id)},
        headers={"Authorization": f"Bearer {viewer_token}"},
        json={
            "team_id": str(team.id),
            "title": "Should fail",
            "source_type": "policy",
            "text": "x" * 100,
        },
    )

    assert response.status_code == 403
