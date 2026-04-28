from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.config import settings
from app.core.redis import get_redis
from app.models.user import User


async def register_user(
    email: str, password: str, name: str | None, db: AsyncSession
) -> User:
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise ValueError("Email already registered")
    user = User(email=email, password_hash=security.hash_password(password), name=name)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(email: str, password: str, db: AsyncSession) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash:
        raise ValueError("Invalid credentials")
    if not security.verify_password(password, user.password_hash):
        raise ValueError("Invalid credentials")
    if not user.is_active:
        raise ValueError("Account inactive")
    return user


async def create_tokens(user_id: str) -> tuple[str, str]:
    access_token = security.create_access_token(user_id)
    refresh_token, jti = security.create_refresh_token(user_id)
    r = await get_redis()
    ttl = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    await r.set(f"refresh:{jti}", user_id, ex=ttl)
    return access_token, refresh_token


async def refresh_access_token(refresh_token: str) -> tuple[str, str]:
    try:
        payload = security.decode_token(refresh_token)
    except JWTError:
        raise ValueError("Invalid refresh token")
    if payload.get("type") != "refresh":
        raise ValueError("Invalid token type")
    jti = payload["jti"]
    user_id = payload["sub"]
    r = await get_redis()
    stored = await r.get(f"refresh:{jti}")
    if not stored:
        raise ValueError("Refresh token expired or revoked")
    await r.delete(f"refresh:{jti}")
    return await create_tokens(user_id)
