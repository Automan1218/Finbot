import pytest
from httpx import ASGITransport, AsyncClient

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
async def test_register(client):
    resp = await client.post("/auth/register", json={
        "email": "alice@example.com",
        "password": "secret123",
        "name": "Alice",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_register_duplicate(client):
    await client.post("/auth/register", json={"email": "bob@example.com", "password": "pw"})
    resp = await client.post("/auth/register", json={"email": "bob@example.com", "password": "pw2"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_login(client):
    await client.post("/auth/register", json={"email": "carol@example.com", "password": "mypass"})
    resp = await client.post("/auth/login", json={"email": "carol@example.com", "password": "mypass"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/auth/register", json={"email": "dave@example.com", "password": "correct"})
    resp = await client.post("/auth/login", json={"email": "dave@example.com", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me(client):
    reg = await client.post("/auth/register", json={
        "email": "eve@example.com",
        "password": "pw",
        "name": "Eve",
    })
    token = reg.json()["access_token"]
    resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == "eve@example.com"
    assert resp.json()["name"] == "Eve"


@pytest.mark.asyncio
async def test_me_unauthorized(client):
    resp = await client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh(client):
    reg = await client.post("/auth/register", json={"email": "frank@example.com", "password": "pw"})
    refresh_token = reg.json()["refresh_token"]
    resp = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    assert "access_token" in resp.json()
    assert "refresh_token" in resp.json()


@pytest.mark.asyncio
async def test_refresh_token_rotation(client):
    reg = await client.post("/auth/register", json={"email": "grace@example.com", "password": "pw"})
    refresh1 = reg.json()["refresh_token"]
    await client.post("/auth/refresh", json={"refresh_token": refresh1})
    resp2 = await client.post("/auth/refresh", json={"refresh_token": refresh1})
    assert resp2.status_code == 401
