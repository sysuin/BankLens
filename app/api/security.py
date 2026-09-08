"""
Passwords and access tokens.

bcrypt for password hashes (the `bcrypt` package directly; passlib is
unmaintained and breaks against bcrypt 4+). HS256 JWTs via PyJWT carrying the
three things every request needs: who (sub), which bank (tenant), and what
they may do (role). Tokens are short-lived by default (8 hours) and there is
no refresh flow in Phase 1 — an RM logs in each morning.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.config import settings

ALGORITHM = "HS256"


class TokenError(Exception):
    """Raised for any token that cannot be trusted."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(
    *,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    tenant_slug: str,
    role: str,
    email: str,
    expires_minutes: int | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "tenant": tenant_slug,
        "role": role,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(
            (
                now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
            ).timestamp()
        ),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("invalid token") from exc
    for key in ("sub", "tenant_id", "tenant", "role"):
        if key not in payload:
            raise TokenError(f"token missing {key}")
    return payload


def assert_secret_is_safe_for(env: str) -> None:
    """Refuse the dev secret outside development. Called at API startup."""
    if env.strip().lower() == "production" and settings.jwt_secret.startswith(
        "dev-only-change-me"
    ):
        raise RuntimeError("JWT_SECRET is the development default; refusing to start.")
