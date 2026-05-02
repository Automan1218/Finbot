import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.auth import service
from app.core.database import get_db
from app.main import app


@pytest.fixture
async def client(db_session):
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_oauth_login_rejects_unsupported_provider(client):
    resp = await client.get("/auth/oauth/not-real")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Unsupported OAuth provider"


@pytest.mark.asyncio
async def test_oauth_login_rejects_unconfigured_provider(client):
    resp = await client.get("/auth/oauth/google")

    assert resp.status_code == 400
    assert "not configured" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_upsert_oauth_user_creates_and_updates(db_session):
    suffix = uuid.uuid4().hex[:8]
    user = await service.upsert_oauth_user(
        provider="github",
        oauth_id=f"gh-{suffix}",
        email=f"oauth_{suffix}@test.com",
        name="OAuth User",
        avatar_url="https://example.com/a.png",
        db=db_session,
    )

    updated = await service.upsert_oauth_user(
        provider="github",
        oauth_id=f"gh-{suffix}",
        email=f"oauth_{suffix}@test.com",
        name="Updated OAuth User",
        avatar_url="https://example.com/b.png",
        db=db_session,
    )

    assert user.id == updated.id
    assert updated.password_hash is None
    assert updated.oauth_provider == "github"
    assert updated.oauth_id == f"gh-{suffix}"
    assert updated.name == "Updated OAuth User"
