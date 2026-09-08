"""
FastAPI dependencies: who is calling, for which bank, with what role.

`current_user()` decodes the bearer token and puts tenant + user into the
request context (so logs and the product validator see them). `require_role`
gates a route to one or more roles. Neither touches the database: the token
is the session, and it expires.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.security import TokenError, decode_access_token
from app.core import context
from app.core.config import settings
from app.platform import ratelimit

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    tenant: str
    role: str
    email: str


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

    principal = Principal(
        user_id=uuid.UUID(payload["sub"]),
        tenant_id=uuid.UUID(payload["tenant_id"]),
        tenant=payload["tenant"],
        role=payload["role"],
        email=payload.get("email", ""),
    )
    # Middleware already set a request id; add tenant and user to the same
    # context so every log line for this request names them.
    context.set_request_context(
        tenant=principal.tenant,
        user=principal.email,
        request_id=context.current_request_id() or context.new_request_id(),
    )
    request.state.principal = principal
    await _rate_limit(principal)
    return principal


# ── Rate limiting ────────────────────────────────────────────────────────────
#
# The backend (in-memory sliding window, or a Postgres counter shared by every
# API process) lives in app/platform/ratelimit.py. This is the one place the
# API calls it. The limit is a setting so tests can lower it.


async def _rate_limit(principal: Principal) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    retry_after = await ratelimit.get_limiter().hit(principal.user_id, limit)
    if retry_after is not None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"rate limit: {limit} requests per minute per user",
            headers={"Retry-After": str(retry_after)},
        )


def reset_rate_limits() -> None:
    """Clear the in-memory limiter and forget the backend choice (tests)."""
    limiter = ratelimit._limiter
    if isinstance(limiter, ratelimit.MemoryLimiter):
        limiter._windows.clear()
    ratelimit.reset_limiter()


def require_role(*roles: str):
    async def _check(principal: Principal = Depends(current_user)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role '{principal.role}' may not do this; needs one of {sorted(roles)}",
            )
        return principal

    return _check
