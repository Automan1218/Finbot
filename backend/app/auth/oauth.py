import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request, status
from starlette.responses import RedirectResponse

from app.core.config import settings

SUPPORTED_PROVIDERS = {"google", "github"}


def _provider_config(provider: str) -> dict[str, str]:
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unsupported OAuth provider",
        )
    if provider == "google":
        return {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
            "scope": "openid email profile",
        }
    return {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "emails_url": "https://api.github.com/user/emails",
        "scope": "read:user user:email",
    }


def _require_configured(provider: str) -> dict[str, str]:
    config = _provider_config(provider)
    if not config["client_id"] or not config["client_secret"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth provider {provider} is not configured",
        )
    return config


async def authorize_redirect(provider: str, request: Request) -> RedirectResponse:
    config = _require_configured(provider)
    redirect_uri = str(request.url_for("oauth_callback")).split("?")[0]
    redirect_uri = f"{redirect_uri}?provider={provider}"
    params = {
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": config["scope"],
        "state": secrets.token_urlsafe(24),
    }
    return RedirectResponse(f"{config['authorize_url']}?{urlencode(params)}")


async def authorize_and_fetch_profile(provider: str, request: Request) -> dict[str, Any]:
    config = _require_configured(provider)
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth callback missing code",
        )

    redirect_uri = str(request.url_for("oauth_callback")).split("?")[0]
    redirect_uri = f"{redirect_uri}?provider={provider}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        token = await _exchange_code(client, config, code, redirect_uri)
        if provider == "google":
            return await _fetch_google_profile(client, config, token)
        return await _fetch_github_profile(client, config, token)


async def _exchange_code(
    client: httpx.AsyncClient,
    config: dict[str, str],
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    response = await client.post(
        config["token_url"],
        data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        headers={"Accept": "application/json"},
    )
    response.raise_for_status()
    return response.json()


async def _fetch_google_profile(
    client: httpx.AsyncClient,
    config: dict[str, str],
    token: dict[str, Any],
) -> dict[str, Any]:
    response = await client.get(
        config["userinfo_url"],
        headers={"Authorization": f"Bearer {token['access_token']}"},
    )
    response.raise_for_status()
    userinfo = response.json()
    return {
        "oauth_id": str(userinfo["sub"]),
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "avatar_url": userinfo.get("picture"),
    }


async def _fetch_github_profile(
    client: httpx.AsyncClient,
    config: dict[str, str],
    token: dict[str, Any],
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token['access_token']}",
        "Accept": "application/vnd.github+json",
    }
    response = await client.get(config["userinfo_url"], headers=headers)
    response.raise_for_status()
    user_data = response.json()
    email = user_data.get("email")
    if not email:
        email_response = await client.get(config["emails_url"], headers=headers)
        email_response.raise_for_status()
        emails = email_response.json()
        primary = next(
            (item for item in emails if item.get("primary") and item.get("verified")),
            None,
        )
        email = primary["email"] if primary else None
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth provider did not return a verified email",
        )
    return {
        "oauth_id": str(user_data["id"]),
        "email": email,
        "name": user_data.get("name") or user_data.get("login"),
        "avatar_url": user_data.get("avatar_url"),
    }
