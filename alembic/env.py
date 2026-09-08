"""
Alembic environment for BankLens.

Runs synchronously over psycopg as the owning role. The URL comes from
settings (DATABASE_ADMIN_URL, converted from the asyncpg driver) or, when
unset, from the embedded local cluster. `BANKLENS_ALEMBIC_URL` overrides both,
which is how the test suite points migrations at a throwaway cluster.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.core.config import settings
from app.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _sync_url() -> str:
    override = os.environ.get("BANKLENS_ALEMBIC_URL")
    if override:
        return override
    if settings.database_admin_url:
        return settings.database_admin_url.replace("+asyncpg", "+psycopg")
    from app.db.local import local_sync_admin_url

    return local_sync_admin_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _sync_url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
