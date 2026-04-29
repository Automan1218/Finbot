import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.team import Team, TeamMember


async def create_team(name: str, owner_id: uuid.UUID, db: AsyncSession) -> Team:
    team = Team(name=name, owner_id=owner_id)
    db.add(team)
    await db.flush()
    member = TeamMember(team_id=team.id, user_id=owner_id, role="owner")
    db.add(member)
    await db.commit()
    await db.refresh(team)
    return team


async def get_team(team_id: uuid.UUID, db: AsyncSession) -> Team | None:
    result = await db.execute(select(Team).where(Team.id == team_id))
    return result.scalar_one_or_none()


async def list_teams_for_user(user_id: uuid.UUID, db: AsyncSession) -> list[Team]:
    result = await db.execute(
        select(Team)
        .join(TeamMember, Team.id == TeamMember.team_id)
        .where(TeamMember.user_id == user_id)
    )
    return list(result.scalars().all())


async def update_team(team_id: uuid.UUID, fields: dict[str, Any], db: AsyncSession) -> Team:
    result = await db.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise ValueError("Team not found")
    for key, val in fields.items():
        setattr(team, key, val)
    await db.commit()
    await db.refresh(team)
    return team


async def delete_team(team_id: uuid.UUID, db: AsyncSession) -> None:
    result = await db.execute(select(Team).where(Team.id == team_id))
    team = result.scalar_one_or_none()
    if not team:
        raise ValueError("Team not found")
    await db.execute(delete(TeamMember).where(TeamMember.team_id == team_id))
    await db.delete(team)
    await db.commit()


async def get_member_role(team_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> str | None:
    result = await db.execute(
        select(TeamMember.role).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def list_members(team_id: uuid.UUID, db: AsyncSession) -> list[TeamMember]:
    result = await db.execute(select(TeamMember).where(TeamMember.team_id == team_id))
    return list(result.scalars().all())


async def add_member(
    team_id: uuid.UUID, user_id: uuid.UUID, role: str, db: AsyncSession
) -> TeamMember:
    member = TeamMember(team_id=team_id, user_id=user_id, role=role)
    db.add(member)
    await db.commit()
    await db.refresh(member)
    return member


async def update_member_role(
    team_id: uuid.UUID, user_id: uuid.UUID, role: str, db: AsyncSession
) -> TeamMember:
    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise ValueError("Member not found")
    member.role = role
    await db.commit()
    await db.refresh(member)
    return member


async def remove_member(team_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> None:
    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()
    if not member:
        raise ValueError("Member not found")
    await db.delete(member)
    await db.commit()
