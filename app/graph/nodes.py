"""
Nodes of the decision graph.

Order: load_context → verify_income → [await_review] → retrieve → narrate →
guardrails → finalize. Every node writes an audit row. The only node that
calls a model is `narrate`; the only node that waits for a person is
`await_review`, and it does so through LangGraph's `interrupt`, which means
the run is checkpointed in Postgres and can be resumed hours later, in a
different process, by `Command(resume=...)`.

A note on `await_review`: LangGraph re-executes an interrupted node from the
top when it resumes, so nothing with a side effect happens in that node
before the interrupt call. The "review requested" audit row is written by
verify_income, which runs once.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt
from sqlalchemy import select

from app.core.config import settings
from app.core.context import tenant_scope
from app.core.logger import get_logger
from app.db.models import (
    Customer,
    Decision,
    DecisionStatus,
    Profile,
    RunStatus,
    Statement,
    StatementMetrics,
    StatementStatus,
)
from app.db.session import tenant_session
from app.graph import audit
from app.graph.state import GraphState
from app.pipeline.analyzer import FinancialMetrics

logger = get_logger(__name__)

INCOME_VERIFICATION = "income_verification"

from app.pipeline.policy import CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT  # noqa: E402


def months_in_period(period: str) -> int:
    """'2024-03' → 1; '2024-01 to 2024-12' → 12."""
    found = re.findall(r"(\d{4})-(\d{2})", period)
    if not found:
        return 1
    (y1, m1), (y2, m2) = found[0], found[-1]
    return max(1, (int(y2) - int(y1)) * 12 + (int(m2) - int(m1)) + 1)


def discrepancy(declared: float, observed: float) -> float:
    if declared <= 0:
        return 100.0 if observed > 0 else 0.0
    return abs(observed - declared) / declared * 100.0


# ── 1. load_context ───────────────────────────────────────────────────────────


async def load_context(state: GraphState) -> dict[str, Any]:
    async with audit.timed() as t:
        async with tenant_session(state["tenant_id"]) as session:
            statement = await session.get(Statement, uuid.UUID(state["statement_id"]))
            if statement is None:
                raise ValueError("statement not visible in this tenant")
            customer = await session.get(Customer, statement.customer_id)
            metrics_row = (
                await session.execute(
                    select(StatementMetrics).where(
                        StatementMetrics.statement_id == statement.id
                    )
                )
            ).scalar_one_or_none()
            if metrics_row is None:
                raise ValueError("statement has no metrics")
            metrics = dict(metrics_row.metrics_json)
            declared = float(customer.declared_monthly_income)
            customer_ref = customer.external_ref
            period = statement.period

    months = months_in_period(period)
    observed = float(metrics["total_income"]) / months

    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.running,
        current_node="load_context",
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="load_context",
        event="loaded",
        actor=state["actor"],
        inputs=audit.inputs_hash(metrics, declared),
        duration_ms=t["ms"],
        payload={
            "period": period,
            "months_covered": months,
            "risk_band": metrics.get("risk_profile"),
            "health_score": metrics.get("financial_health_score"),
        },
    )
    return {
        "metrics": metrics,
        "customer_ref": customer_ref,
        "declared_monthly_income": declared,
        "observed_monthly_income": round(observed, 2),
        "months_covered": months,
    }


# ── 2. verify_income ──────────────────────────────────────────────────────────


async def verify_income(state: GraphState) -> dict[str, Any]:
    declared = state["declared_monthly_income"]
    observed = state["observed_monthly_income"]
    pct = round(discrepancy(declared, observed), 2)
    review_threshold = settings.income_review_threshold_pct
    log_threshold = settings.income_log_threshold_pct
    review_required = pct > review_threshold

    decision_id = uuid.uuid4()
    async with tenant_session(state["tenant_id"]) as session:
        session.add(
            Decision(
                id=decision_id,
                tenant_id=uuid.UUID(state["tenant_id"]),
                run_id=uuid.UUID(state["run_id"]),
                statement_id=uuid.UUID(state["statement_id"]),
                kind=INCOME_VERIFICATION,
                declared_monthly_income=declared,
                observed_monthly_income=observed,
                discrepancy_pct=pct,
                threshold_pct=review_threshold,
                status=(
                    DecisionStatus.pending
                    if review_required
                    else DecisionStatus.auto_cleared
                ),
                requested_by=None,
                note=(
                    None
                    if review_required
                    else (
                        f"within threshold ({pct:.1f}% <= {review_threshold:.0f}%)"
                        + ("; above log threshold" if pct > log_threshold else "")
                    )
                ),
            )
        )

    outcome = "review_requested" if review_required else "auto_cleared"
    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.awaiting_review if review_required else RunStatus.running,
        current_node="verify_income",
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="verify_income",
        event=outcome,
        actor=state["actor"],
        inputs=audit.inputs_hash(declared, observed),
        payload={
            "declared_monthly_income": declared,
            "observed_monthly_income": observed,
            "discrepancy_pct": pct,
            "review_threshold_pct": review_threshold,
            "log_threshold_pct": log_threshold,
            "decision_id": str(decision_id),
        },
    )
    logger.info(
        "verify_income run=%s declared=%.2f observed=%.2f pct=%.1f -> %s",
        state["run_id"],
        declared,
        observed,
        pct,
        outcome,
    )
    return {
        "discrepancy_pct": pct,
        "review_required": review_required,
        "decision_id": str(decision_id),
        "review_decision": None if review_required else "not_required",
    }


def route_after_verify(state: GraphState) -> str:
    return "await_review" if state.get("review_required") else "retrieve"


# ── 3. await_review (interrupt) ───────────────────────────────────────────────


async def await_review(state: GraphState) -> dict[str, Any]:
    # Everything above this line runs again on resume; keep it pure.
    answer = interrupt(
        {
            "kind": INCOME_VERIFICATION,
            "decision_id": state["decision_id"],
            "declared_monthly_income": state["declared_monthly_income"],
            "observed_monthly_income": state["observed_monthly_income"],
            "discrepancy_pct": state["discrepancy_pct"],
            "customer_ref": state.get("customer_ref"),
        }
    )
    # From here on we are resuming with the reviewer's answer.
    action = str(answer.get("action", "")).lower()
    if action not in {"approved", "rejected"}:
        raise ValueError(f"review answer must be approved or rejected, got {action!r}")
    reviewer = str(answer.get("reviewer", "unknown"))
    note = answer.get("note")

    async with tenant_session(state["tenant_id"]) as session:
        decision = await session.get(Decision, uuid.UUID(state["decision_id"]))
        if decision is not None:
            decision.status = (
                DecisionStatus.approved
                if action == "approved"
                else DecisionStatus.rejected
            )
            decision.decided_at = datetime.now(timezone.utc)
            decision.note = note
            reviewer_id = answer.get("reviewer_id")
            if reviewer_id:
                decision.decided_by = uuid.UUID(str(reviewer_id))

    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="await_review",
        event=f"review_{action}",
        actor=reviewer,
        payload={"decision_id": state["decision_id"], "note": note},
    )
    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.running,
        current_node="await_review",
    )
    return {"review_decision": action, "review_note": note, "reviewer": reviewer}


def route_after_review(state: GraphState) -> str:
    return (
        "retrieve"
        if state.get("review_decision") == "approved"
        else "finalize_rejected"
    )


# ── 4. retrieve ───────────────────────────────────────────────────────────────


def _retrieve_sync(metrics: FinancialMetrics, tenant: str) -> list[dict]:
    from langchain_community.callbacks.manager import get_openai_callback

    from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve
    from app.platform.tracing import set_llm_usage, span

    with tenant_scope(tenant):
        with span("rag.build_vector_store"):
            store = build_vector_store(tenant)
        with span("rag.retrieve") as current:
            with get_openai_callback() as cb:
                chunks = retrieve(build_retrieval_query(metrics), store, tenant=tenant)
            # Multi-query rewrites run on the mini model; embeddings are
            # fractions of a cent and are not counted here.
            if cb.prompt_tokens or cb.completion_tokens:
                set_llm_usage(
                    settings.openai_mini_model, cb.prompt_tokens, cb.completion_tokens
                )
            current.set_attribute("rag.sources", sorted({c["source"] for c in chunks}))
            current.set_attribute("rag.candidates", len(chunks))
        return chunks


async def retrieve(state: GraphState) -> dict[str, Any]:
    metrics = FinancialMetrics.model_validate(state["metrics"])
    async with audit.timed() as t:
        chunks = await asyncio.to_thread(_retrieve_sync, metrics, state["tenant"])
    await audit.set_run_status(
        state["tenant_id"], state["run_id"], RunStatus.running, current_node="retrieve"
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="retrieve",
        event="retrieved",
        actor="system",
        inputs=audit.inputs_hash(state["metrics"]),
        duration_ms=t["ms"],
        payload={
            "sources": sorted({c["source"] for c in chunks}),
            "count": len(chunks),
        },
    )
    return {"chunks": chunks}


# ── 5. narrate ────────────────────────────────────────────────────────────────


def _narrate_sync(
    metrics: FinancialMetrics, chunks: list[dict], tenant: str
) -> tuple[dict, bool, int, int]:
    from langchain_community.callbacks.manager import get_openai_callback

    from app.api.service import prompt_version
    from app.pipeline.cache import cached_build_profile
    from app.platform.tracing import set_llm_usage, span

    with tenant_scope(tenant):
        with span("llm.profile") as current:
            with get_openai_callback() as cb:
                profile, from_cache = cached_build_profile(
                    metrics, chunks, tenant=tenant
                )
            set_llm_usage(
                settings.openai_model,
                cb.prompt_tokens,
                cb.completion_tokens,
                prompt_version=prompt_version(),
            )
            current.set_attribute("llm.cache_hit", from_cache)
    return (
        profile.model_dump(mode="json"),
        from_cache,
        cb.prompt_tokens,
        cb.completion_tokens,
    )


async def narrate(state: GraphState) -> dict[str, Any]:
    from app.api.service import prompt_version

    from app.platform import registry

    metrics = FinancialMetrics.model_validate(state["metrics"])
    async with audit.timed() as t:
        profile, from_cache, tokens_in, tokens_out = await asyncio.to_thread(
            _narrate_sync, metrics, state["chunks"], state["tenant"]
        )
    if not from_cache:
        await registry.register_use(settings.openai_model)
    await audit.set_run_status(
        state["tenant_id"], state["run_id"], RunStatus.running, current_node="narrate"
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="narrate",
        event="cache_hit" if from_cache else "generated",
        actor="system",
        inputs=audit.inputs_hash(state["metrics"], state["chunks"]),
        model=settings.openai_model,
        prompt_version=prompt_version(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=t["ms"],
        payload={
            "primary_product": profile.get("primary_product"),
            "secondary_product": profile.get("secondary_product"),
        },
    )
    return {
        "profile": profile,
        "from_cache": from_cache,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


# ── 6. guardrails ─────────────────────────────────────────────────────────────


async def guardrails(state: GraphState) -> dict[str, Any]:
    from app.pipeline.agent import resolve_product

    tenant = state["tenant"]
    profile = state["profile"]
    metrics = state["metrics"]
    violations: list[str] = []
    warnings: list[str] = []

    retrieved = {c["source"] for c in state.get("chunks", [])}
    forbidden = CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT.get(tenant, frozenset())
    for field in ("primary_product", "secondary_product"):
        name = profile.get(field, "")
        resolved = resolve_product(name, tenant)
        if resolved is None:
            violations.append(f"{field} '{name}' is not in the {tenant} catalogue")
            continue
        if metrics.get("is_cashflow_negative") and resolved in forbidden:
            violations.append(
                f"{field} '{name}' is unsecured credit; customer is in cash-flow deficit"
            )
        if resolved not in retrieved:
            warnings.append(f"{field} '{name}' was not among the retrieved sources")

    # The narrative must not carry an injected instruction or a PII shape out.
    from app.platform.guardrails import scan_output

    for field in (
        "income_stability_analysis",
        "spending_pattern_breakdown",
        "credit_risk_assessment",
        "primary_reason",
        "secondary_reason",
    ):
        verdict = scan_output(str(profile.get(field, "")))
        if verdict.blocked:
            violations.append(
                f"{field} carries {'/'.join(verdict.families)} content: "
                f"{'; '.join(verdict.matched)}"
            )

    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.running,
        current_node="guardrails",
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="guardrails",
        event="blocked" if violations else "passed",
        actor="system",
        payload={"violations": violations, "warnings": warnings},
    )
    return {"guardrail_violations": violations, "guardrail_warnings": warnings}


def route_after_guardrails(state: GraphState) -> str:
    return "finalize_blocked" if state.get("guardrail_violations") else "finalize"


# ── 7. finalize ───────────────────────────────────────────────────────────────


async def finalize(state: GraphState) -> dict[str, Any]:
    from app.api.service import prompt_version

    profile_id = uuid.uuid4()
    async with tenant_session(state["tenant_id"]) as session:
        statement = await session.get(Statement, uuid.UUID(state["statement_id"]))
        session.add(
            Profile(
                id=profile_id,
                tenant_id=uuid.UUID(state["tenant_id"]),
                statement_id=uuid.UUID(state["statement_id"]),
                model=settings.openai_model,
                prompt_version=prompt_version(),
                retrieved_sources=sorted({c["source"] for c in state["chunks"]}),
                profile_json=state["profile"],
                from_cache=state.get("from_cache", False),
            )
        )
        if statement is not None:
            statement.status = StatementStatus.profiled

    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.completed,
        current_node="finalize",
        profile_id=profile_id,
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="finalize",
        event="completed",
        actor="system",
        payload={
            "profile_id": str(profile_id),
            "review_decision": state.get("review_decision"),
            "warnings": state.get("guardrail_warnings", []),
        },
    )
    return {"profile_id": str(profile_id), "outcome": "completed"}


async def finalize_rejected(state: GraphState) -> dict[str, Any]:
    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.rejected,
        current_node="finalize_rejected",
        error=f"income verification rejected: {state.get('review_note') or 'no note'}",
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="finalize_rejected",
        event="rejected",
        actor=state.get("reviewer") or "system",
        payload={"note": state.get("review_note")},
    )
    return {"outcome": "rejected"}


async def finalize_blocked(state: GraphState) -> dict[str, Any]:
    reason = "; ".join(state.get("guardrail_violations", []))
    await audit.set_run_status(
        state["tenant_id"],
        state["run_id"],
        RunStatus.failed,
        current_node="finalize_blocked",
        error=f"guardrail: {reason}",
    )
    await audit.record(
        state["tenant_id"],
        run_id=state["run_id"],
        statement_id=state["statement_id"],
        node="finalize_blocked",
        event="blocked",
        actor="system",
        payload={"violations": state.get("guardrail_violations", [])},
    )
    return {"outcome": "blocked", "error": reason}
