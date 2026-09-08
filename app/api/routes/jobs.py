"""
Bulk work: enqueue statements for the worker, watch them finish.

    POST /jobs           multipart: customer_id, kind (ingest | ingest_and_run), file
    GET  /jobs           this tenant's jobs, newest first (status, duration, cost)
    GET  /jobs/{id}
    GET  /jobs/summary   throughput and cost for the tenant's finished jobs

Also the gateway's view of the world:

    GET  /platform/gateway   providers, circuit state, budget spent today,
                             and which provider the next call would use
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, select

from app.api.deps import Principal, current_user, require_role
from app.db.models import Customer, Job, JobStatus
from app.db.session import tenant_session
from app.platform import gateway

router = APIRouter(tags=["jobs"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
KINDS = {"ingest", "ingest_and_run"}


class JobOut(BaseModel):
    id: uuid.UUID
    kind: str
    customer_id: uuid.UUID
    filename: str
    status: str
    attempts: int
    worker: str | None
    statement_id: uuid.UUID | None
    run_id: uuid.UUID | None
    error: str | None
    duration_ms: int | None
    cost_usd: float | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def _out(job: Job) -> JobOut:
    return JobOut(
        id=job.id,
        kind=job.kind,
        customer_id=job.customer_id,
        filename=job.filename,
        status=job.status.value,
        attempts=job.attempts,
        worker=job.worker,
        statement_id=job.statement_id,
        run_id=job.run_id,
        error=job.error,
        duration_ms=job.duration_ms,
        cost_usd=float(job.cost_usd) if job.cost_usd is not None else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.post("/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def enqueue(
    customer_id: uuid.UUID = Form(...),
    kind: str = Form("ingest"),
    file: UploadFile = File(...),
    principal: Principal = Depends(require_role("rm")),
) -> JobOut:
    if kind not in KINDS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"kind must be one of {sorted(KINDS)}"
        )
    content = await file.read()
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "file empty or over 5 MB"
        )
    async with tenant_session(principal.tenant_id) as session:
        if await session.get(Customer, customer_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "customer not found")
        job = Job(
            id=uuid.uuid4(),
            tenant_id=principal.tenant_id,
            tenant_slug=principal.tenant,
            kind=kind,
            customer_id=customer_id,
            requested_by=principal.user_id,
            requested_by_email=principal.email,
            filename=(file.filename or "statement")[:255],
            content=content,
            status=JobStatus.queued,
        )
        session.add(job)
        await session.flush()
        await session.refresh(job)
        return _out(job)


@router.get("/jobs/summary")
async def summary(principal: Principal = Depends(current_user)) -> dict:
    async with tenant_session(principal.tenant_id) as session:
        rows = (
            await session.execute(
                select(
                    Job.status,
                    func.count(),
                    func.coalesce(func.avg(Job.duration_ms), 0),
                    func.coalesce(func.sum(Job.cost_usd), 0),
                ).group_by(Job.status)
            )
        ).all()
        finished = (
            await session.execute(
                select(
                    func.min(Job.started_at), func.max(Job.finished_at), func.count()
                ).where(Job.status.in_([JobStatus.done, JobStatus.awaiting_review]))
            )
        ).one()
    by_status = {
        s.value: {
            "count": int(n),
            "avg_duration_ms": float(avg),
            "cost_usd": float(cost),
        }
        for s, n, avg, cost in rows
    }
    first, last, n = finished
    wall_s = (last - first).total_seconds() if first and last and n else 0.0
    return {
        "by_status": by_status,
        "finished": int(n or 0),
        "wall_seconds": round(wall_s, 1),
        "statements_per_minute": round(n / wall_s * 60, 1) if wall_s > 0 else None,
    }


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(principal: Principal = Depends(current_user)) -> list[JobOut]:
    async with tenant_session(principal.tenant_id) as session:
        rows = (
            (
                await session.execute(
                    select(Job).order_by(Job.created_at.desc()).limit(500)
                )
            )
            .scalars()
            .all()
        )
        return [_out(j) for j in rows]


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(
    job_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> JobOut:
    async with tenant_session(principal.tenant_id) as session:
        job = await session.get(Job, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return _out(job)


@router.get("/platform/gateway")
async def gateway_status(principal: Principal = Depends(current_user)) -> dict:
    import asyncio

    return await asyncio.to_thread(gateway.status, principal.tenant)
