"""
The bulk worker: takes jobs off the queue and runs them.

    python -m app.worker                 # run until stopped
    python -m app.worker --once          # drain the queue, then exit
    python -m app.worker --max-jobs 50   # process at most N, then exit

The queue is a Postgres table. A worker claims a job with
`SELECT ... FOR UPDATE SKIP LOCKED`, so many workers can run at once without
handing out the same job twice, and sets a lease; a job whose lease expired
(its worker died) is claimable again. Claiming runs as the owner role
because the worker serves every tenant; the work itself runs inside the
tenant's session, so RLS applies to everything the job touches.

Cost and duration per job come from the spans the job produced, which is
the same source the numbers card and the tenant budget use.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, text

from app.core.config import settings
from app.core.context import tenant_scope
from app.core.logger import get_logger
from app.db.models import Job, JobStatus
from app.db.session import admin_session, dispose_engines, tenant_session
from app.platform import gateway
from app.platform.tracing import (
    ATTR_STATEMENT_ID,
    ATTR_TENANT,
    ATTR_TENANT_ID,
    setup_tracing,
    shutdown_tracing,
    span,
)

logger = get_logger(__name__)
WORKER_NAME = f"{socket.gethostname()}:{os.getpid()}"


async def claim_job() -> Job | None:
    """Claim the oldest queued (or lease-expired) job, or None."""
    lease = timedelta(seconds=settings.job_lease_seconds)
    async with admin_session() as session:
        row = (
            await session.execute(
                select(Job)
                .where(
                    (Job.status == JobStatus.queued)
                    | (
                        (Job.status == JobStatus.running)
                        & (Job.leased_until < datetime.now(timezone.utc))
                    )
                )
                .order_by(Job.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.status = JobStatus.running
        row.attempts += 1
        row.worker = WORKER_NAME
        row.leased_until = datetime.now(timezone.utc) + lease
        row.started_at = datetime.now(timezone.utc)
        await session.flush()
        job_id = row.id
    async with admin_session() as session:
        return await session.get(Job, job_id)


async def _cost_for(tenant_id: uuid.UUID, statement_id: uuid.UUID | None) -> float:
    if statement_id is None:
        return 0.0
    async with tenant_session(tenant_id) as session:
        total = (
            await session.execute(
                text(
                    # Every span in every trace that touched the statement:
                    # the categorizer's model calls are children of ingest
                    # and carry no statement id of their own.
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM spans WHERE trace_id IN "
                    "(SELECT DISTINCT trace_id FROM spans WHERE statement_id = :s)"
                ),
                {"s": str(statement_id)},
            )
        ).scalar_one()
    return float(total or 0)


async def process(job: Job) -> None:
    """Run one job: ingest, optionally the graph. Records outcome and cost."""
    from app.api.service import ingest_statement_file
    from app.graph.builder import start_run

    started = time.perf_counter()
    statement_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    final = JobStatus.done
    error: str | None = None
    try:
        with tenant_scope(job.tenant_slug, user=job.requested_by_email):
            with span(
                "job.process",
                **{
                    ATTR_TENANT: job.tenant_slug,
                    ATTR_TENANT_ID: str(job.tenant_id),
                    "job.id": str(job.id),
                    "job.kind": job.kind,
                },
            ) as job_span:
                statement_id = await ingest_statement_file(
                    tenant_id=job.tenant_id,
                    tenant_slug=job.tenant_slug,
                    customer_id=job.customer_id,
                    uploaded_by=job.requested_by,
                    filename=job.filename,
                    content=bytes(job.content),
                    uploaded_by_email=job.requested_by_email,
                )
                job_span.set_attribute(ATTR_STATEMENT_ID, str(statement_id))
                if job.kind == "ingest_and_run":
                    async for event in start_run(
                        tenant=job.tenant_slug,
                        tenant_id=job.tenant_id,
                        statement_id=statement_id,
                        actor_email=job.requested_by_email,
                        actor_id=job.requested_by,
                    ):
                        if event.get("event") == "run":
                            run_id = uuid.UUID(event["run_id"])
                        elif event.get("event") == "interrupt":
                            final = JobStatus.awaiting_review
                        elif event.get("event") == "error":
                            final = JobStatus.failed
                            error = str(event.get("data"))
    except Exception as exc:  # noqa: BLE001 - the job records its own failure
        logger.exception("job %s failed", job.id)
        final = JobStatus.failed
        error = f"{type(exc).__name__}: {exc}"

    duration_ms = int((time.perf_counter() - started) * 1000)
    cost = await _cost_for(job.tenant_id, statement_id)
    async with admin_session() as session:
        row = await session.get(Job, job.id)
        row.status = final
        row.error = error
        row.statement_id = statement_id
        row.run_id = run_id
        row.duration_ms = duration_ms
        row.cost_usd = cost
        row.finished_at = datetime.now(timezone.utc)
        row.leased_until = None
    logger.info(
        "job %s %s in %d ms cost=$%.4f statement=%s",
        job.id,
        final.value,
        duration_ms,
        cost,
        statement_id,
    )


async def run_worker(once: bool = False, max_jobs: int | None = None) -> int:
    """Claim and process jobs with bounded concurrency. Returns jobs processed."""
    setup_tracing()
    gateway.enable_budgets()
    logger.info(
        "worker %s starting concurrency=%d", WORKER_NAME, settings.worker_concurrency
    )
    processed = 0
    semaphore = asyncio.Semaphore(settings.worker_concurrency)
    in_flight: set[asyncio.Task] = set()

    async def _run(job: Job) -> None:
        async with semaphore:
            await process(job)

    try:
        while True:
            if max_jobs is not None and processed >= max_jobs:
                break
            job = (
                await claim_job()
                if len(in_flight) < settings.worker_concurrency
                else None
            )
            if job is None:
                if in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    continue
                if once:
                    break
                await asyncio.sleep(1.0)
                continue
            processed += 1
            in_flight.add(asyncio.create_task(_run(job)))
        if in_flight:
            await asyncio.gather(*in_flight)
    finally:
        from app.graph.builder import close_checkpointer

        await close_checkpointer()
        await dispose_engines()
        shutdown_tracing()
    logger.info("worker %s stopping after %d job(s)", WORKER_NAME, processed)
    return processed


def main() -> int:
    ap = argparse.ArgumentParser(description="BankLens bulk worker")
    ap.add_argument("--once", action="store_true", help="drain the queue, then exit")
    ap.add_argument("--max-jobs", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run_worker(once=args.once, max_jobs=args.max_jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
