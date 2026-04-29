import uuid

import pytest
import pytest_asyncio
from app.core.security import hash_password
from app.models.team import Team, TeamMember
from app.models.user import User
from app.teams import service


@pytest_asyncio.fixture
async def owner(db_session):
    u = User(email=f"owner_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw"))
    db_session.add(u)
    await db_session.commit()
    await db_session.refresh(u)
    return u


@pytest_asyncio.fixture
async def other_user(db_session):
    u = User(email=f"other_{uuid.uuid4().hex[:6]}@t.com", password_hash=hash_password("pw"))
    db_session.add(u)
    await db_session.commit()
    await db_session.refresh(u)
    return u


@pytest.mark.asyncio
async def test_create_team_adds_owner_member(db_session, owner):
    team = await service.create_team("Alpha", owner.id, db_session)
    assert team.name == "Alpha"
    assert team.owner_id == owner.id
    role = await service.get_member_role(team.id, owner.id, db_session)
    assert role == "owner"


@pytest.mark.asyncio
async def test_list_teams_for_user(db_session, owner):
    await service.create_team("T1", owner.id, db_session)
    await service.create_team("T2", owner.id, db_session)
    teams = await service.list_teams_for_user(owner.id, db_session)
    names = [t.name for t in teams]
    assert "T1" in names and "T2" in names


@pytest.mark.asyncio
async def test_update_team(db_session, owner):
    team = await service.create_team("OldName", owner.id, db_session)
    updated = await service.update_team(team.id, {"name": "NewName"}, db_session)
    assert updated.name == "NewName"


@pytest.mark.asyncio
async def test_add_and_remove_member(db_session, owner, other_user):
    team = await service.create_team("TeamX", owner.id, db_session)
    member = await service.add_member(team.id, other_user.id, "member", db_session)
    assert member.role == "member"

    await service.remove_member(team.id, other_user.id, db_session)
    role = await service.get_member_role(team.id, other_user.id, db_session)
    assert role is None


@pytest.mark.asyncio
async def test_update_member_role(db_session, owner, other_user):
    team = await service.create_team("TeamY", owner.id, db_session)
    await service.add_member(team.id, other_user.id, "member", db_session)
    updated = await service.update_member_role(team.id, other_user.id, "admin", db_session)
    assert updated.role == "admin"


@pytest.mark.asyncio
async def test_list_members(db_session, owner, other_user):
    team = await service.create_team("TeamZ", owner.id, db_session)
    await service.add_member(team.id, other_user.id, "viewer", db_session)
    members = await service.list_members(team.id, db_session)
    assert len(members) == 2  # owner + other_user


@pytest.mark.asyncio
async def test_update_nonexistent_member_raises(db_session, owner):
    team = await service.create_team("TeamW", owner.id, db_session)
    with pytest.raises(ValueError, match="Member not found"):
        await service.update_member_role(team.id, uuid.uuid4(), "admin", db_session)


@pytest.mark.asyncio
async def test_remove_nonexistent_member_raises(db_session, owner):
    team = await service.create_team("TeamV", owner.id, db_session)
    with pytest.raises(ValueError, match="Member not found"):
        await service.remove_member(team.id, uuid.uuid4(), db_session)
