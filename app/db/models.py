"""
BankLens platform schema (SQLAlchemy 2.0 declarative).

Every table that holds customer data carries `tenant_id` and is protected by a
Postgres row-level-security policy (see alembic/versions/0001_platform.py).
The policy compares tenant_id with the session variable `app.tenant_id`,
which `app.db.session.tenant_session()` sets inside every transaction. The
API's database role is subject to the policies; the owning role used by
migrations and the seed script is not.

Phase 1 tables: tenants, users, customers, statements, transactions,
statement_metrics, profiles. Phase 2 adds decisions, approvals and the audit
table on top of these without changing them.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Role(str, enum.Enum):
    rm = "rm"
    reviewer = "reviewer"


class SourceType(str, enum.Enum):
    csv = "csv"
    pdf = "pdf"


class StatementStatus(str, enum.Enum):
    analyzed = "analyzed"
    profiled = "profiled"


class RunStatus(str, enum.Enum):
    running = "running"
    awaiting_review = "awaiting_review"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"


class DecisionStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    auto_cleared = "auto_cleared"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TenantScoped:
    """Mixin: every row belongs to exactly one tenant."""

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )


class User(TenantScoped, Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role, name="user_role"), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Customer(TenantScoped, Base):
    __tablename__ = "customers"
    __table_args__ = (
        UniqueConstraint("tenant_id", "external_ref", name="uq_customers_tenant_ref"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    external_ref: Mapped[str] = mapped_column(String(64), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    # What the customer told the bank at onboarding. Phase 2 compares this
    # with the income observed in the statement (income verification).
    declared_monthly_income: Mapped[float] = mapped_column(
        Numeric(14, 2), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    statements: Mapped[list["Statement"]] = relationship(back_populates="customer")


class Statement(TenantScoped, Base):
    __tablename__ = "statements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, name="statement_source"), nullable=False
    )
    period: Mapped[str] = mapped_column(String(64), nullable=False)
    transaction_count: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[StatementStatus] = mapped_column(
        Enum(StatementStatus, name="statement_status"),
        nullable=False,
        default=StatementStatus.analyzed,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer: Mapped[Customer] = relationship(back_populates="statements")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="statement", cascade="all, delete-orphan"
    )
    metrics: Mapped["StatementMetrics | None"] = relationship(
        back_populates="statement", uselist=False, cascade="all, delete-orphan"
    )
    profiles: Mapped[list["Profile"]] = relationship(
        back_populates="statement", cascade="all, delete-orphan"
    )


class Transaction(TenantScoped, Base):
    """One sanitized, categorized ledger line. Phase 6 queries this table."""

    __tablename__ = "transactions"
    __table_args__ = (Index("ix_transactions_statement", "statement_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=False,
    )
    txn_date: Mapped[date] = mapped_column(Date, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    amount: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    txn_type: Mapped[str] = mapped_column(String(16), nullable=False)  # Credit / Debit
    category: Mapped[str] = mapped_column(String(64), nullable=False)

    statement: Mapped[Statement] = relationship(back_populates="transactions")


class StatementMetrics(TenantScoped, Base):
    """The deterministic numbers, stored as columns so SQL can reach them."""

    __tablename__ = "statement_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    statement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    total_income: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    total_expenses: Mapped[float] = mapped_column(Numeric(14, 2), nullable=False)
    savings_rate_pct: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    expense_to_income_ratio: Mapped[float] = mapped_column(
        Numeric(8, 4), nullable=False
    )
    risk_band: Mapped[str] = mapped_column(String(16), nullable=False)
    health_score: Mapped[int] = mapped_column(Integer, nullable=False)
    # The full FinancialMetrics payload, so the UI and chat rebuild the
    # exact object the pipeline produced without recomputation drift.
    metrics_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    statement: Mapped[Statement] = relationship(back_populates="metrics")


class Profile(TenantScoped, Base):
    """One LLM-generated profile for a statement, with what produced it."""

    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    statement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieved_sources: Mapped[list] = mapped_column(JSONB, nullable=False)
    profile_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    from_cache: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    statement: Mapped[Statement] = relationship(back_populates="profiles")


class Run(TenantScoped, Base):
    """One execution of the decision graph. `id` is the LangGraph thread id."""

    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_statement", "statement_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    statement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=False,
    )
    started_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), nullable=False, default=RunStatus.running
    )
    current_node: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Decision(TenantScoped, Base):
    """
    A decision with consequences: income verification.

    Created by the graph's verify_income node. `pending` rows are the review
    queue; a reviewer's approve/reject resumes the graph from its checkpoint.
    """

    __tablename__ = "decisions"
    __table_args__ = (Index("ix_decisions_tenant_status", "tenant_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False
    )
    statement_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    declared_monthly_income: Mapped[float] = mapped_column(
        Numeric(14, 2), nullable=False
    )
    observed_monthly_income: Mapped[float] = mapped_column(
        Numeric(14, 2), nullable=False
    )
    discrepancy_pct: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    threshold_pct: Mapped[float] = mapped_column(Numeric(8, 2), nullable=False)
    status: Mapped[DecisionStatus] = mapped_column(
        Enum(DecisionStatus, name="decision_status"), nullable=False
    )
    requested_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class AuditEvent(TenantScoped, Base):
    """
    Append-only trail: one row per graph node (and per human action).

    Answers "who decided what, on which inputs, with which prompt and model".
    Inputs are recorded as a hash so the trail never duplicates PII.
    """

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_run", "run_id"),
        Index("ix_audit_statement", "statement_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=True
    )
    statement_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("statements.id", ondelete="CASCADE"),
        nullable=True,
    )
    node: Mapped[str] = mapped_column(String(64), nullable=False)
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(320), nullable=False)
    inputs_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# Tables whose rows are tenant-owned and therefore carry an RLS policy.
TENANT_TABLES = (
    "users",
    "customers",
    "statements",
    "transactions",
    "statement_metrics",
    "profiles",
    "runs",
    "decisions",
    "audit_events",
)
