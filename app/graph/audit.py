"""
The audit trail: one row per graph node, per human action.

Rows are written inside the tenant's transaction (RLS applies) and the API
role can only INSERT and SELECT on audit_events, so nothing the graph or a
route does can rewrite history. Inputs are recorded as a SHA-256 of their
canonical JSON, never as the values, so the trail can be retained for years
without becoming a second copy of customer data.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.db.models import AuditEvent, Run, RunStatus
from app.db.session import tenant_session


def inputs_hash(*parts: Any) -> str:
    canonical = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


async def record(
    tenant_id: str | uuid.UUID,
    *,
    node: str,
    event: str,
    actor: str,
    run_id: str | uuid.UUID | None = None,
    statement_id: str | uuid.UUID | None = None,
    payload: dict | None = None,
    inputs: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    duration_ms: int | None = None,
) -> None:
    async with tenant_session(tenant_id) as session:
        session.add(
            AuditEvent(
                tenant_id=uuid.UUID(str(tenant_id)),
                run_id=uuid.UUID(str(run_id)) if run_id else None,
                statement_id=uuid.UUID(str(statement_id)) if statement_id else None,
                node=node,
                event=event,
                actor=actor,
                inputs_hash=inputs,
                model=model,
                prompt_version=prompt_version,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
                payload=payload or {},
            )
        )


@asynccontextmanager
async def timed() -> AsyncIterator[dict]:
    """`async with timed() as t:` then read t["ms"] afterwards."""
    box: dict = {"ms": 0}
    started = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = int((time.perf_counter() - started) * 1000)


async def set_run_status(
    tenant_id: str | uuid.UUID,
    run_id: str | uuid.UUID,
    status: RunStatus,
    *,
    current_node: str | None = None,
    profile_id: str | uuid.UUID | None = None,
    error: str | None = None,
) -> None:
    async with tenant_session(tenant_id) as session:
        run = await session.get(Run, uuid.UUID(str(run_id)))
        if run is None:
            return
        run.status = status
        if current_node is not None:
            run.current_node = current_node
        if profile_id is not None:
            run.profile_id = uuid.UUID(str(profile_id))
        if error is not None:
            run.error = error[:2000]
