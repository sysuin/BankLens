"""Request and response models for the BankLens API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

# ── Auth ─────────────────────────────────────────────────────────────────────


class LoginRequest(BaseModel):
    tenant: str = Field(description="Tenant slug, e.g. 'meridian'")
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant: str
    role: str
    email: str
    full_name: str


class MeResponse(BaseModel):
    user_id: uuid.UUID
    tenant: str
    tenant_name: str
    role: str
    email: str
    full_name: str


# ── Customers ────────────────────────────────────────────────────────────────


class CustomerCreate(BaseModel):
    external_ref: str = Field(max_length=64)
    full_name: str = Field(max_length=200)
    declared_monthly_income: float = Field(ge=0)


class CustomerOut(BaseModel):
    id: uuid.UUID
    external_ref: str
    full_name: str
    declared_monthly_income: float
    statement_count: int = 0


# ── Statements ───────────────────────────────────────────────────────────────


class TransactionOut(BaseModel):
    date: str
    description: str
    amount: float
    type: str
    category: str


class StatementSummary(BaseModel):
    id: uuid.UUID
    customer_id: uuid.UUID
    customer_name: str
    filename: str
    source_type: str
    period: str
    transaction_count: int
    status: str
    risk_band: str | None = None
    health_score: int | None = None
    savings_rate_pct: float | None = None
    guardrail_flags: dict = Field(default_factory=dict)
    created_at: datetime


class StatementDetail(StatementSummary):
    metrics: dict
    transactions: list[TransactionOut]
    declared_monthly_income: float


class ProfileOut(BaseModel):
    id: uuid.UUID
    statement_id: uuid.UUID
    model: str
    prompt_version: str
    retrieved_sources: list[str]
    from_cache: bool
    profile: dict
    created_at: datetime


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


# ── Decision graph ───────────────────────────────────────────────────────────


class RunOut(BaseModel):
    id: uuid.UUID
    statement_id: uuid.UUID
    status: str
    current_node: str | None = None
    profile_id: uuid.UUID | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    pending_decision_id: uuid.UUID | None = None


class DecisionOut(BaseModel):
    id: uuid.UUID
    run_id: uuid.UUID
    statement_id: uuid.UUID
    customer_ref: str
    customer_name: str
    kind: str
    declared_monthly_income: float
    observed_monthly_income: float
    discrepancy_pct: float
    threshold_pct: float
    status: str
    note: str | None = None
    decided_at: datetime | None = None
    created_at: datetime


class ReviewRequest(BaseModel):
    action: str = Field(pattern="^(approve|reject)$")
    note: str | None = Field(default=None, max_length=2000)


class AuditEventOut(BaseModel):
    id: int
    run_id: uuid.UUID | None
    statement_id: uuid.UUID | None
    node: str
    event: str
    actor: str
    inputs_hash: str | None
    model: str | None
    prompt_version: str | None
    tokens_in: int | None
    tokens_out: int | None
    duration_ms: int | None
    payload: dict
    created_at: datetime


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    database: str
    tenants_on_disk: list[str]
    version: str
