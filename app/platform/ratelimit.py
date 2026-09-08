"""
Per-user request rate limiting, with a backend per deployment shape.

    memory    one process: a sliding minute in a dict. The original.
    postgres  any number of processes: a fixed-minute counter row per user,
              incremented with one upsert on the API role. No Redis needed;
              the same database that already holds the queue and the cache
              holds a few hundred bytes per active user per minute.

Both implement `Limiter.hit(user_id, limit) -> int | None`: the seconds to
wait if the caller is over the limit, otherwise None. The dependency in
`app/api/deps.py` picks one from `settings.rate_limit_backend`. A Redis
backend would be a third class with the same two methods.
"""

from __future__ import annotations

import collections
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Limiter(Protocol):
    async def hit(self, user_id: uuid.UUID, limit: int) -> int | None: ...

    async def reset(self) -> None: ...


class MemoryLimiter:
    """Sliding one-minute window per user, in process memory."""

    def __init__(self) -> None:
        self._windows: dict[uuid.UUID, collections.deque] = {}
        self._lock = threading.Lock()

    async def hit(self, user_id: uuid.UUID, limit: int) -> int | None:
        now = time.monotonic()
        with self._lock:
            window = self._windows.setdefault(user_id, collections.deque())
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= limit:
                return 60
            window.append(now)
        return None

    async def reset(self) -> None:
        with self._lock:
            self._windows.clear()


_UPSERT = text("""
    INSERT INTO rate_limit_buckets (user_id, bucket, count)
    VALUES (:user_id, :bucket, 1)
    ON CONFLICT (user_id, bucket)
    DO UPDATE SET count = rate_limit_buckets.count + 1
    RETURNING count
    """)
_SWEEP = text("DELETE FROM rate_limit_buckets WHERE bucket < :cutoff")


class PostgresLimiter:
    """
    Fixed one-minute buckets per user, shared by every API process.

    One upsert per request, on the API role. Old buckets are swept
    opportunistically (about one request in a hundred) so the table stays
    at "active users x 2 rows". A fixed window admits at most 2x the limit
    across a boundary, which is the usual trade for one round trip.
    """

    def __init__(self, engine_factory) -> None:
        self._engine_factory = engine_factory
        self._hits = 0

    async def hit(self, user_id: uuid.UUID, limit: int) -> int | None:
        now = datetime.now(timezone.utc)
        bucket = now.replace(second=0, microsecond=0)
        engine: AsyncEngine = self._engine_factory()
        async with engine.begin() as conn:
            count = (
                await conn.execute(_UPSERT, {"user_id": user_id, "bucket": bucket})
            ).scalar_one()
            self._hits += 1
            if self._hits % 100 == 0:
                await conn.execute(_SWEEP, {"cutoff": bucket - timedelta(minutes=2)})
        if count > limit:
            return max(1, 60 - now.second)
        return None

    async def reset(self) -> None:
        engine: AsyncEngine = self._engine_factory()
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM rate_limit_buckets"))


_limiter: Limiter | None = None
_limiter_backend: str | None = None


def get_limiter() -> Limiter:
    """The process-wide limiter for the configured backend (rebuilt on change)."""
    global _limiter, _limiter_backend
    backend = settings.rate_limit_backend
    if _limiter is None or _limiter_backend != backend:
        if backend == "postgres":
            from app.db.session import app_engine

            _limiter = PostgresLimiter(app_engine)
        else:
            _limiter = MemoryLimiter()
        _limiter_backend = backend
        logger.info("rate limiter backend: %s", backend)
    return _limiter


def reset_limiter() -> None:
    """Forget the limiter (tests, engine resets)."""
    global _limiter, _limiter_backend
    _limiter = _limiter_backend = None
