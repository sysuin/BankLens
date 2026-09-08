"""
Database sessions with the tenant pinned for row-level security.

Two engines, two roles:

    app role    (settings.database_url)        used by the API. Subject to RLS.
    admin role  (settings.database_admin_url)  owns the tables; migrations and
                                               the seed script only.

`tenant_session(tenant_id)` opens a transaction and runs
`SET LOCAL app.tenant_id = '<uuid>'` before handing the session out. Because
it is SET LOCAL, the setting dies with the transaction; a pooled connection
cannot leak one tenant's scope into the next request. Every RLS policy reads
that variable. A session opened without it sees zero tenant rows — which is
the failure mode you want when someone forgets.

When neither URL is configured, both resolve to the embedded local Postgres
started by `make db` (see app.db.local).
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_app_engine: AsyncEngine | None = None
_admin_engine: AsyncEngine | None = None
_chat_engine: AsyncEngine | None = None


def _resolve_urls() -> tuple[str, str]:
    """(app_url, admin_url), falling back to the embedded local cluster."""
    app_url, admin_url = settings.database_url, settings.database_admin_url
    if app_url and admin_url:
        return app_url, admin_url
    from app.db.local import ensure_local_cluster, local_urls

    # Falling back to the embedded cluster: make sure it is actually up.
    # Idempotent, and it is what lets `make api` work on a fresh laptop.
    ensure_local_cluster()
    local_app, local_admin = local_urls()
    return app_url or local_app, admin_url or local_admin


def app_engine() -> AsyncEngine:
    global _app_engine
    if _app_engine is None:
        url, _ = _resolve_urls()
        _app_engine = create_async_engine(url, pool_pre_ping=True, pool_size=5)
    return _app_engine


def chat_engine() -> AsyncEngine:
    """The read-only warehouse role: tenant-filtered views only."""
    global _chat_engine
    if _chat_engine is None:
        url = settings.database_chat_url
        if not url:
            from app.db.local import ensure_local_cluster, local_chat_url

            ensure_local_cluster()
            url = local_chat_url()
        _chat_engine = create_async_engine(url, pool_pre_ping=True, pool_size=3)
    return _chat_engine


def admin_engine() -> AsyncEngine:
    global _admin_engine
    if _admin_engine is None:
        _, url = _resolve_urls()
        _admin_engine = create_async_engine(url, pool_pre_ping=True, pool_size=2)
    return _admin_engine


def reset_engines() -> None:
    """
    Forget the engines without disposing them.

    For code that is about to run on a different event loop (the worker
    started from a test, a script after the API): asyncpg connections are
    bound to the loop that created them, so the next caller must build new
    engines rather than reuse these.
    """
    global _app_engine, _admin_engine, _chat_engine
    _app_engine = _admin_engine = _chat_engine = None


async def dispose_engines() -> None:
    global _app_engine, _admin_engine, _chat_engine
    for engine in (_app_engine, _admin_engine, _chat_engine):
        if engine is not None:
            await engine.dispose()
    _app_engine = _admin_engine = _chat_engine = None


def _sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def tenant_session(
    tenant_id: uuid.UUID | str, engine: AsyncEngine | None = None
) -> AsyncIterator[AsyncSession]:
    """
    A transaction on the app role with `app.tenant_id` pinned.

    Commits on clean exit, rolls back on error. The SET LOCAL is the tenant
    filter: remove it and every tenant table reads as empty. `engine` lets a
    worker or a test bring its own (bound to its own event loop).
    """
    async with _sessionmaker(engine or app_engine())() as session:
        async with session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tenant, true)"),
                {"tenant": str(tenant_id)},
            )
            yield session


@asynccontextmanager
async def unscoped_app_session(
    engine: AsyncEngine | None = None,
) -> AsyncIterator[AsyncSession]:
    """
    App-role transaction with NO tenant pinned.

    Only legitimate use: resolving a tenant by slug at login, which reads the
    `tenants` table (no RLS). Reading any tenant table here returns nothing.
    """
    async with _sessionmaker(engine or app_engine())() as session:
        async with session.begin():
            yield session


@asynccontextmanager
async def admin_session(
    engine: AsyncEngine | None = None,
) -> AsyncIterator[AsyncSession]:
    """Owner-role transaction. Bypasses RLS. Migrations and seeding only."""
    async with _sessionmaker(engine or admin_engine())() as session:
        async with session.begin():
            yield session
