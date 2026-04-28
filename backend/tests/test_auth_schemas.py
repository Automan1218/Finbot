import pytest
from pydantic import ValidationError
from app.auth.schemas import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserResponse


def test_register_request_valid():
    r = RegisterRequest(email="user@example.com", password="secret", name="Alice")
    assert r.email == "user@example.com"
    assert r.name == "Alice"


def test_register_request_invalid_email():
    with pytest.raises(ValidationError):
        RegisterRequest(email="not-an-email", password="secret")


def test_register_request_name_optional():
    r = RegisterRequest(email="x@example.com", password="pw")
    assert r.name is None


def test_token_response_default_type():
    t = TokenResponse(access_token="a", refresh_token="r")
    assert t.token_type == "bearer"
