import pytest
from app.auth.service import authenticate_user, create_tokens, refresh_access_token, register_user


@pytest.mark.asyncio
async def test_register_user(db_session):
    user = await register_user("svc_alice@test.com", "pass123", "Alice", db_session)
    assert user.id is not None
    assert user.email == "svc_alice@test.com"
    assert user.password_hash != "pass123"


@pytest.mark.asyncio
async def test_register_duplicate_raises(db_session):
    await register_user("svc_dup@test.com", "pw", None, db_session)
    with pytest.raises(ValueError, match="already registered"):
        await register_user("svc_dup@test.com", "pw2", None, db_session)


@pytest.mark.asyncio
async def test_authenticate_user(db_session):
    await register_user("svc_bob@test.com", "correct", None, db_session)
    user = await authenticate_user("svc_bob@test.com", "correct", db_session)
    assert user.email == "svc_bob@test.com"


@pytest.mark.asyncio
async def test_authenticate_wrong_password(db_session):
    await register_user("svc_carol@test.com", "right", None, db_session)
    with pytest.raises(ValueError, match="Invalid credentials"):
        await authenticate_user("svc_carol@test.com", "wrong", db_session)


@pytest.mark.asyncio
async def test_create_and_refresh_tokens():
    access, refresh = await create_tokens("test-user-id")
    assert access
    assert refresh
    new_access, new_refresh = await refresh_access_token(refresh)
    assert new_access != access
    assert new_refresh != refresh


@pytest.mark.asyncio
async def test_get_current_user_valid(db_session):
    from app.auth.dependencies import get_current_user
    from app.core.security import create_access_token
    from fastapi.security import HTTPAuthorizationCredentials

    user = await register_user("dep_alice@test.com", "pw", None, db_session)
    token = create_access_token(str(user.id))
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    result = await get_current_user(credentials=creds, db=db_session)
    assert result.id == user.id


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(db_session):
    from app.auth.dependencies import get_current_user
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad.token.here")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials=creds, db=db_session)
    assert exc_info.value.status_code == 401
