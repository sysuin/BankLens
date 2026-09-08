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
    _rate_limit(principal)
    return principal


# ── Rate limiting ────────────────────────────────────────────────────────────
#
# Sliding one-minute window per user, in process memory. Correct for one API
# process; a multi-process deployment swaps `_windows` for Redis and keeps
# this function's shape. The limit is a setting so tests can lower it.

import collections  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

_windows: dict[uuid.UUID, collections.deque] = {}
_windows_lock = threading.Lock()


def _rate_limit(principal: Principal) -> None:
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    with _windows_lock:
        window = _windows.setdefault(principal.user_id, collections.deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                f"rate limit: {limit} requests per minute per user",
                headers={"Retry-After": "60"},
            )
        window.append(now)


def reset_rate_limits() -> None:
    with _windows_lock:
        _windows.clear()


def require_role(*roles: str):
    async def _check(principal: Principal = Depends(current_user)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role '{principal.role}' may not do this; needs one of {sorted(roles)}",
            )
        return principal

    return _check
