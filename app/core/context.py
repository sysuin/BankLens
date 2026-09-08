"""
Request-scoped context for BankLens.

Three values travel with every request without being threaded through every
function signature: the tenant, the acting user, and a request id. They live
in contextvars so they are visible to the logger, the product-name validator
in agent.py, and any pipeline stage that needs to know whose data it is on.

Set by the API middleware and dependencies; set explicitly by scripts, the
MCP server and the Streamlit direct mode via `tenant_scope()`.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from typing import Iterator

from app.core.config import settings

_tenant: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "banklens_tenant", default=None
)
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "banklens_request_id", default=None
)
_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "banklens_user", default=None
)


def current_tenant() -> str:
    """The tenant slug in scope, falling back to the configured default."""
    return _tenant.get() or settings.default_tenant


def current_request_id() -> str | None:
    return _request_id.get()


def current_user() -> str | None:
    return _user.get()


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def tenant_scope(
    tenant: str, *, user: str | None = None, request_id: str | None = None
) -> Iterator[None]:
    """Run a block with the given tenant (and optionally user/request) in scope."""
    tokens = [
        _tenant.set(tenant),
        _user.set(user),
        _request_id.set(request_id or new_request_id()),
    ]
    try:
        yield
    finally:
        for var, token in zip((_tenant, _user, _request_id), tokens):
            var.reset(token)


def set_request_context(
    *, tenant: str | None, user: str | None, request_id: str
) -> list[contextvars.Token]:
    """Middleware-style setter; returns tokens for `reset_request_context`."""
    return [_tenant.set(tenant), _user.set(user), _request_id.set(request_id)]


def reset_request_context(tokens: list[contextvars.Token]) -> None:
    for var, token in zip((_tenant, _user, _request_id), tokens):
        var.reset(token)
