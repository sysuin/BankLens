"""
Shared fixtures.

The platform tests need a real Postgres because row-level security is the
thing under test. `pg_cluster` boots an embedded cluster (pgserver) in a
temporary directory, runs the Alembic migration as the owner, seeds the two
synthetic banks, and points the app's engines at it. It is torn down and
deleted at the end of the session. No Docker, no network.

Pipeline tests (test_rag, test_agent, ...) do not use these fixtures and stay
as fast as before.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def pg_cluster(tmp_path_factory):
    """A migrated, seeded, throwaway Postgres. Yields ids from the seed."""
    from app.core.config import settings
    from app.db import local, session as db_session

    pgdata = tmp_path_factory.mktemp("pg", numbered=False)
    server = local.ensure_local_cluster(pgdata, persistent=False)

    app_url, admin_url = local.local_urls(pgdata)
    settings.database_url = app_url
    settings.database_admin_url = admin_url

    env = dict(os.environ, BANKLENS_ALEMBIC_URL=local.local_sync_admin_url(pgdata))
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )

    from app.db.seed import seed

    ids = asyncio.run(seed(with_statements=False))
    asyncio.run(db_session.dispose_engines())

    yield {"ids": ids, "pgdata": pgdata, "app_url": app_url, "admin_url": admin_url}

    asyncio.run(db_session.dispose_engines())
    server.cleanup()


@pytest.fixture()
def client(pg_cluster):
    """A TestClient over the real app, with lifespan, against the cluster."""
    from fastapi.testclient import TestClient

    from app.api.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def login(client, tenant: str, local: str = "rm") -> dict:
    """Log in as a seeded user; returns headers with the bearer token."""
    response = client.post(
        "/auth/login",
        json={
            "tenant": tenant,
            "email": f"{local}@{tenant}.example",
            "password": "banklens-demo",
        },
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture()
def meridian_rm(client):
    return login(client, "meridian", "rm")


@pytest.fixture()
def meridian_reviewer(client):
    return login(client, "meridian", "reviewer")


@pytest.fixture()
def harbor_rm(client):
    return login(client, "harbor", "rm")


SAMPLE_CSV = ROOT / "data" / "sample_1_high_saver.csv"


def run_db(url: str, fn):
    """
    Run `await fn(engine)` on a fresh event loop with its own engine.

    The app's engines belong to the TestClient's loop; asyncpg connections
    cannot cross loops, so database-level tests bring their own.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _go():
        engine = create_async_engine(url)
        try:
            return await fn(engine)
        finally:
            await engine.dispose()

    return asyncio.run(_go())


def upload_sample(
    client, headers: dict, customer_id: str, path: Path = SAMPLE_CSV
) -> dict:
    with path.open("rb") as fh:
        response = client.post(
            "/statements",
            headers=headers,
            data={"customer_id": customer_id},
            files={"file": (path.name, fh, "text/csv")},
        )
    assert response.status_code == 201, response.text
    return response.json()
