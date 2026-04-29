import uuid

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.core.security import create_access_token, hash_password
from app.main import app
from app.models.team import Team, TeamMember
from app.models.user import User


@pytest_asyncio.fixture
async def teams_client(db_session):
    u = User(email=f"tc_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw"))
    db_session.add(u)
    await db_session.commit()
    await db_session.refresh(u)
    token = create_access_token(str(u.id))

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", headers=headers
    ) as c:
        yield c, u, db_session
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_create_team(teams_client):
    c, user, _ = teams_client
    resp = await c.post("/teams", json={"name": "Beta"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "Beta"
    assert resp.json()["owner_id"] == str(user.id)


@pytest.mark.asyncio
async def test_list_teams(teams_client):
    c, _, _ = teams_client
    await c.post("/teams", json={"name": "ListA"})
    await c.post("/teams", json={"name": "ListB"})
    resp = await c.get("/teams")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()]
    assert "ListA" in names and "ListB" in names


@pytest.mark.asyncio
async def test_get_team(teams_client):
    c, _, _ = teams_client
    create = await c.post("/teams", json={"name": "GetMe"})
    tid = create.json()["id"]
    resp = await c.get(f"/teams/{tid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "GetMe"


@pytest.mark.asyncio
async def test_update_team_owner_only(teams_client):
    c, _, _ = teams_client
    create = await c.post("/teams", json={"name": "OldN"})
    tid = create.json()["id"]
    resp = await c.patch(f"/teams/{tid}", json={"name": "NewN"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "NewN"


@pytest.mark.asyncio
async def test_delete_team(teams_client):
    c, _, _ = teams_client
    create = await c.post("/teams", json={"name": "DelMe"})
    tid = create.json()["id"]
    resp = await c.delete(f"/teams/{tid}")
    assert resp.status_code == 204
    get_resp = await c.get(f"/teams/{tid}")
    assert get_resp.status_code == 403


@pytest.mark.asyncio
async def test_list_members(teams_client):
    c, _, _ = teams_client
    create = await c.post("/teams", json={"name": "MembersT"})
    tid = create.json()["id"]
    resp = await c.get(f"/teams/{tid}/members")
    assert resp.status_code == 200
    assert len(resp.json()) == 1  # just the owner


@pytest.mark.asyncio
async def test_add_and_remove_member(teams_client):
    c, owner, db_session = teams_client
    invitee = User(
        email=f"inv_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw")
    )
    db_session.add(invitee)
    await db_session.commit()
    await db_session.refresh(invitee)

    create = await c.post("/teams", json={"name": "InviteT"})
    tid = create.json()["id"]

    add_resp = await c.post(
        f"/teams/{tid}/members", json={"email": invitee.email, "role": "member"}
    )
    assert add_resp.status_code == 201
    assert add_resp.json()["role"] == "member"

    members_resp = await c.get(f"/teams/{tid}/members")
    assert len(members_resp.json()) == 2

    del_resp = await c.delete(f"/teams/{tid}/members/{invitee.id}")
    assert del_resp.status_code == 204

    members_resp2 = await c.get(f"/teams/{tid}/members")
    assert len(members_resp2.json()) == 1


@pytest.mark.asyncio
async def test_update_member_role(teams_client):
    c, owner, db_session = teams_client
    invitee = User(
        email=f"rol_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw")
    )
    db_session.add(invitee)
    await db_session.commit()
    await db_session.refresh(invitee)

    create = await c.post("/teams", json={"name": "RoleT"})
    tid = create.json()["id"]
    await c.post(f"/teams/{tid}/members", json={"email": invitee.email, "role": "member"})

    resp = await c.patch(
        f"/teams/{tid}/members/{invitee.id}/role", json={"role": "admin"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


@pytest.mark.asyncio
async def test_non_member_cannot_access_team(teams_client):
    c, owner, db_session = teams_client
    other = User(email=f"other_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw"))
    db_session.add(other)
    await db_session.flush()
    other_team = Team(name="OtherTeam", owner_id=other.id)
    db_session.add(other_team)
    await db_session.flush()
    db_session.add(TeamMember(team_id=other_team.id, user_id=other.id, role="owner"))
    await db_session.commit()

    resp = await c.get(f"/teams/{other_team.id}")
    assert resp.status_code == 403
