"""
Prove the tenant guard is real by breaking it on purpose.

    make prove-isolation        (or: python -m scripts.prove_isolation)

On a throwaway embedded Postgres: migrate, seed both banks, store one
Meridian statement, then read it three times as the API role pinned to
Harbor:

    1. RLS on       -> nothing comes back            (expected)
    2. RLS disabled -> the Meridian row leaks         (the guard was doing the work)
    3. RLS back on  -> nothing again                  (expected)

Nothing here touches the development database.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def main() -> int:
    os.chdir(ROOT)
    from app.core.config import settings
    from app.db import local
    from app.db import session as db_session
    from app.db.models import Statement
    from app.db.seed import seed
    from app.api.service import ingest_statement_file

    pgdata = Path(tempfile.mkdtemp(prefix="banklens_prove_"))
    server = local.ensure_local_cluster(pgdata, persistent=False)
    app_url, admin_url = local.local_urls(pgdata)
    settings.database_url, settings.database_admin_url = app_url, admin_url

    env = dict(os.environ, BANKLENS_ALEMBIC_URL=local.local_sync_admin_url(pgdata))
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )

    async def setup() -> tuple[dict, uuid.UUID]:
        ids = await seed(with_statements=False)
        sample = ROOT / "data" / "sample_1_high_saver.csv"
        statement_id = await ingest_statement_file(
            tenant_id=ids["tenants"]["meridian"],
            tenant_slug="meridian",
            customer_id=ids["customers"]["M-1001"],
            uploaded_by=ids["users"]["rm@meridian"],
            filename=sample.name,
            content=sample.read_bytes(),
        )
        await db_session.dispose_engines()
        return ids, statement_id

    ids, statement_id = asyncio.run(setup())

    async def read_as_harbor() -> int:
        engine = create_async_engine(app_url)
        try:
            async with db_session.tenant_session(
                ids["tenants"]["harbor"], engine=engine
            ) as session:
                rows = await session.execute(
                    select(Statement).where(Statement.id == statement_id)
                )
                return len(rows.scalars().all())
        finally:
            await engine.dispose()

    async def set_rls(enabled: bool) -> None:
        engine = create_async_engine(admin_url)
        try:
            async with engine.begin() as conn:
                verb = "ENABLE" if enabled else "DISABLE"
                await conn.execute(
                    text(f"ALTER TABLE statements {verb} ROW LEVEL SECURITY")
                )
        finally:
            await engine.dispose()

    def report(step: str, leaked: int, expect_leak: bool) -> bool:
        ok = (leaked > 0) == expect_leak
        colour = GREEN if ok else RED
        outcome = "row LEAKED" if leaked else "nothing returned"
        print(
            f"{colour}{step:<42} {outcome:<18} {'as expected' if ok else 'UNEXPECTED'}{RESET}"
        )
        return ok

    print(f"\nMeridian statement {statement_id} read as Harbor's API session:\n")
    results = [
        report(
            "1. row-level security ON", asyncio.run(read_as_harbor()), expect_leak=False
        )
    ]
    asyncio.run(set_rls(False))
    results.append(
        report(
            "2. row-level security DISABLED",
            asyncio.run(read_as_harbor()),
            expect_leak=True,
        )
    )
    asyncio.run(set_rls(True))
    results.append(
        report(
            "3. row-level security back ON",
            asyncio.run(read_as_harbor()),
            expect_leak=False,
        )
    )

    server.cleanup()
    print()
    if all(results):
        print(
            f"{GREEN}The tenant guard lives in the database, not in a WHERE clause.{RESET}"
        )
        return 0
    print(f"{RED}Isolation proof FAILED.{RESET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
