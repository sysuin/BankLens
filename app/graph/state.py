"""
State carried through the decision graph.

Everything a node needs is in this dict, and every node returns only the
keys it changed. Values are JSON-serialisable on purpose: LangGraph
checkpoints the state to Postgres after every node, and a run that pauses
for a reviewer is resumed, possibly in another process, from exactly this.

Two boundaries are visible here. The numbers (`metrics`, `observed_monthly_
income`, `discrepancy_pct`) are computed in code before any model runs. The
review decision comes from a person and is recorded before the model runs.
The model only fills `profile`, and even that is validated afterwards.
"""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # Identity
    tenant: str
    tenant_id: str
    statement_id: str
    run_id: str
    actor: str  # email of the RM who started the run

    # Deterministic inputs (load_context, verify_income)
    metrics: dict[str, Any]
    customer_ref: str
    declared_monthly_income: float
    observed_monthly_income: float
    months_covered: int
    discrepancy_pct: float
    review_required: bool
    decision_id: str | None

    # Human decision (await_review)
    review_decision: str | None  # "approved" | "rejected"
    review_note: str | None
    reviewer: str | None

    # Probabilistic half (retrieve, narrate)
    chunks: list[dict[str, Any]]
    profile: dict[str, Any]
    from_cache: bool
    tokens_in: int
    tokens_out: int

    # Guardrails and outcome
    guardrail_violations: list[str]
    guardrail_warnings: list[str]
    profile_id: str | None
    outcome: str  # "completed" | "rejected" | "blocked"
    error: str | None
