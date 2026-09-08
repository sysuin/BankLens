"""Read side of the decision graph: runs, the review queue, the audit trail."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.api.service import NotFound
from app.db.models import (
    AuditEvent,
    Customer,
    Decision,
    DecisionStatus,
    Run,
    Statement,
)
from app.db.session import tenant_session


def _run_out(run: Run, pending_decision_id: uuid.UUID | None = None) -> dict:
    return {
        "id": run.id,
        "statement_id": run.statement_id,
        "status": run.status.value,
        "current_node": run.current_node,
        "profile_id": run.profile_id,
        "error": run.error,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "pending_decision_id": pending_decision_id,
    }


async def list_runs(tenant_id: uuid.UUID, statement_id: uuid.UUID) -> list[dict]:
    async with tenant_session(tenant_id) as session:
        statement = await session.get(Statement, statement_id)
        if statement is None:
            raise NotFound("statement not found")
        runs = (
            (
                await session.execute(
                    select(Run)
                    .where(Run.statement_id == statement_id)
                    .order_by(Run.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
        pending = {
            d.run_id: d.id
            for d in (
                await session.execute(
                    select(Decision).where(
                        Decision.statement_id == statement_id,
                        Decision.status == DecisionStatus.pending,
                    )
                )
            )
            .scalars()
            .all()
        }
        return [_run_out(r, pending.get(r.id)) for r in runs]


async def get_run(tenant_id: uuid.UUID, run_id: uuid.UUID) -> dict:
    async with tenant_session(tenant_id) as session:
        run = await session.get(Run, run_id)
        if run is None:
            raise NotFound("run not found")
        pending = (
            await session.execute(
                select(Decision.id).where(
                    Decision.run_id == run_id, Decision.status == DecisionStatus.pending
                )
            )
        ).scalar_one_or_none()
        return _run_out(run, pending)


def _decision_out(d: Decision, customer: Customer) -> dict:
    return {
        "id": d.id,
        "run_id": d.run_id,
        "statement_id": d.statement_id,
        "customer_ref": customer.external_ref,
        "customer_name": customer.full_name,
        "kind": d.kind,
        "declared_monthly_income": float(d.declared_monthly_income),
        "observed_monthly_income": float(d.observed_monthly_income),
        "discrepancy_pct": float(d.discrepancy_pct),
        "threshold_pct": float(d.threshold_pct),
        "status": d.status.value,
        "note": d.note,
        "decided_at": d.decided_at,
        "created_at": d.created_at,
    }


async def list_decisions(
    tenant_id: uuid.UUID, status: DecisionStatus | None = DecisionStatus.pending
) -> list[dict]:
    async with tenant_session(tenant_id) as session:
        query = (
            select(Decision, Customer)
            .join(Statement, Statement.id == Decision.statement_id)
            .join(Customer, Customer.id == Statement.customer_id)
            .order_by(Decision.created_at.desc())
        )
        if status is not None:
            query = query.where(Decision.status == status)
        rows = (await session.execute(query)).all()
        return [_decision_out(d, c) for d, c in rows]


async def get_decision(tenant_id: uuid.UUID, decision_id: uuid.UUID) -> dict:
    async with tenant_session(tenant_id) as session:
        row = (
            await session.execute(
                select(Decision, Customer)
                .join(Statement, Statement.id == Decision.statement_id)
                .join(Customer, Customer.id == Statement.customer_id)
                .where(Decision.id == decision_id)
            )
        ).first()
        if row is None:
            raise NotFound("decision not found")
        d, c = row
        return _decision_out(d, c)


async def list_audit(
    tenant_id: uuid.UUID,
    *,
    statement_id: uuid.UUID | None = None,
    run_id: uuid.UUID | None = None,
    limit: int = 200,
) -> list[dict]:
    async with tenant_session(tenant_id) as session:
        query = select(AuditEvent).order_by(AuditEvent.id.asc()).limit(limit)
        if statement_id is not None:
            statement = await session.get(Statement, statement_id)
            if statement is None:
                raise NotFound("statement not found")
            query = query.where(AuditEvent.statement_id == statement_id)
        if run_id is not None:
            query = query.where(AuditEvent.run_id == run_id)
        rows = (await session.execute(query)).scalars().all()
        return [
            {
                "id": e.id,
                "run_id": e.run_id,
                "statement_id": e.statement_id,
                "node": e.node,
                "event": e.event,
                "actor": e.actor,
                "inputs_hash": e.inputs_hash,
                "model": e.model,
                "prompt_version": e.prompt_version,
                "tokens_in": e.tokens_in,
                "tokens_out": e.tokens_out,
                "duration_ms": e.duration_ms,
                "payload": e.payload,
                "created_at": e.created_at,
            }
            for e in rows
        ]
