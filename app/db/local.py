"""
Embedded local Postgres for development and tests — no Docker required.

`pgserver` ships Postgres binaries inside a wheel and runs a cluster under a
directory of our choosing, listening on a unix socket. `ensure_local_cluster()`
starts it (idempotent), creates the two roles the platform expects, and the
`banklens` database. `local_urls()` returns the SQLAlchemy async URLs the
session module falls back to when DATABASE_URL / DATABASE_ADMIN_URL are empty.

Production and Docker Compose use a real Postgres and set both URLs; this
module is never imported there.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import settings

APP_ROLE = "banklens_app"
CHAT_ROLE = "banklens_chat"
ADMIN_ROLE = "postgres"
DB_NAME = "banklens"


def cluster_dir() -> Path:
    raw = os.environ.get("BANKLENS_LOCAL_PG_DIR") or settings.local_pg_dir
    path = Path(raw).expanduser().resolve()
    if " " in str(path):
        raise ValueError(
            f"Local Postgres directory must not contain spaces: {path}. "
            "Set LOCAL_PG_DIR to a path without spaces."
        )
    return path


def _bootstrap_sql(sock: Path, statements: list[str]) -> None:
    """Run bootstrap statements as the superuser over the socket (autocommit)."""
    import psycopg

    with psycopg.connect(
        host=str(sock), user=ADMIN_ROLE, dbname="postgres", autocommit=True
    ) as conn:
        for statement in statements:
            conn.execute(statement)


def ensure_local_cluster(directory: Path | None = None, *, persistent: bool = True):
    """
    Start (or attach to) the embedded cluster and prepare roles + database.

    persistent=True leaves the server running after this process exits, which
    is what `make db` wants. Tests pass False so the cluster is torn down and
    deleted with the temporary directory.
    """
    import pgserver

    pgdata = directory or cluster_dir()
    pgdata.mkdir(parents=True, exist_ok=True)
    server = pgserver.get_server(
        str(pgdata), cleanup_mode=None if persistent else "delete"
    )

    import psycopg

    with psycopg.connect(
        host=str(pgdata), user=ADMIN_ROLE, dbname="postgres", autocommit=True
    ) as conn:
        has_role = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)
        ).fetchone()
        if not has_role:
            conn.execute(f"CREATE ROLE {APP_ROLE} LOGIN")
        has_chat = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (CHAT_ROLE,)
        ).fetchone()
        if not has_chat:
            conn.execute(f"CREATE ROLE {CHAT_ROLE} LOGIN NOINHERIT")
        has_db = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)
        ).fetchone()
        if not has_db:
            conn.execute(f"CREATE DATABASE {DB_NAME} OWNER {ADMIN_ROLE}")
            conn.execute(f"GRANT CONNECT ON DATABASE {DB_NAME} TO {APP_ROLE}")
            conn.execute(f"GRANT CONNECT ON DATABASE {DB_NAME} TO {CHAT_ROLE}")
    return server


def _socket_dir(directory: Path | None = None) -> Path:
    return directory or cluster_dir()


def local_urls(directory: Path | None = None) -> tuple[str, str]:
    """(app_url, admin_url) for the embedded cluster via its unix socket."""
    sock = _socket_dir(directory)
    app = f"postgresql+asyncpg://{APP_ROLE}@/{DB_NAME}?host={sock}"
    admin = f"postgresql+asyncpg://{ADMIN_ROLE}@/{DB_NAME}?host={sock}"
    return app, admin


def local_chat_url(directory: Path | None = None) -> str:
    """The read-only chat role's URL for the embedded cluster."""
    sock = _socket_dir(directory)
    return f"postgresql+asyncpg://{CHAT_ROLE}@/{DB_NAME}?host={sock}"


def local_sync_admin_url(directory: Path | None = None) -> str:
    """psycopg URL for Alembic, which runs synchronously."""
    sock = _socket_dir(directory)
    return f"postgresql+psycopg://{ADMIN_ROLE}@/{DB_NAME}?host={sock}"


def stop_local_cluster(directory: Path | None = None) -> None:
    """Stop the embedded cluster started by `make db`."""
    from pgserver._commands import pg_ctl

    pgdata = directory or cluster_dir()
    pg_ctl(["-D", str(pgdata), "stop"], pgdata=str(pgdata), user=None, timeout=30)
