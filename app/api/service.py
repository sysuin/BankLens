"""
Service layer: runs the pipeline and persists its output under a tenant.

The pipeline modules are synchronous and CPU/network bound (pdfplumber,
pandas, OpenAI). They run in a worker thread via `asyncio.to_thread` so the
event loop stays free; the database work is async and always happens inside
`tenant_session()`, which pins `app.tenant_id` for row-level security.

Nothing here decides anything. The deterministic numbers come from
analyzer.py; the narrative comes from agent.py; this module stores both and
rebuilds the exact objects the UI and the chat need.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from decimal import Decimal

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.context import tenant_scope
from app.core.logger import get_logger
from app.db.models import (
    Customer,
    Profile,
    SourceType,
    Statement,
    StatementMetrics,
    StatementStatus,
    Transaction,
)
from app.db.session import tenant_session
from app.pipeline.agent import SYSTEM_PROMPT_PATH, CustomerProfile
from app.pipeline.analyzer import FinancialMetrics, compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.sanitizer import sanitize_dataframe

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"date", "description", "amount", "type"}


class NotFound(Exception):
    """A row the caller may not see, or that does not exist. Same answer."""


class BadInput(ValueError):
    """The uploaded file could not be turned into a statement."""


# ── Pipeline (sync, run in a thread) ─────────────────────────────────────────


def _parse_upload(filename: str, content: bytes) -> tuple[pd.DataFrame, SourceType]:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        from app.pipeline.pdf_parser import parse_pdf_statement

        return parse_pdf_statement(io.BytesIO(content)), SourceType.pdf
    if lowered.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [c.strip().lower() for c in df.columns]
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise BadInput(
                f"CSV missing required columns: {sorted(missing)}. "
                "Expected: date, description, amount, type (Credit/Debit)."
            )
        return df, SourceType.csv
    raise BadInput("Only .csv and .pdf statements are accepted.")


def _analyze_sync(filename: str, content: bytes, tenant_slug: str):
    """(categorized df, metrics, source type, guardrail scan)."""
    from app.platform.tracing import span

    with tenant_scope(tenant_slug):
        with span(
            "pipeline.parse", **{"file.kind": filename.rsplit(".", 1)[-1].lower()}
        ):
            raw, source_type = _parse_upload(filename, content)
        with span("pipeline.sanitize"):
            sanitized = sanitize_dataframe(raw)
        with span("guardrail.statement_scan") as current:
            # The statement is attacker-controlled text: neutralise
            # instruction-like descriptions before the categorizer or any
            # model sees them. Numbers are untouched.
            from app.platform.guardrails import scan_statement

            sanitized, scan = scan_statement(sanitized)
            current.set_attribute("guardrail.flagged_rows", scan.flagged_rows)
            current.set_attribute("guardrail.families", sorted(set(scan.families)))
        with span("pipeline.categorize") as current:
            categorized = categorize_dataframe(sanitized)
            current.set_attribute(
                "pipeline.llm_fallback_rows",
                int((categorized["category"] == "Others").sum()),
            )
        with span("pipeline.metrics") as current:
            metrics = compute_metrics(categorized)
            current.set_attribute("metrics.rows", int(metrics.transaction_count))
    return categorized, metrics, source_type, scan


def _profile_sync(
    metrics: FinancialMetrics, tenant_slug: str
) -> tuple[CustomerProfile, list[dict], bool]:
    from app.pipeline.cache import cached_build_profile
    from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve

    from langchain_community.callbacks.manager import get_openai_callback

    from app.platform.tracing import set_llm_usage, span

    with tenant_scope(tenant_slug):
        with span("rag.retrieve"):
            chunks = retrieve(
                build_retrieval_query(metrics),
                build_vector_store(tenant_slug),
                tenant=tenant_slug,
            )
        with span("llm.profile") as current:
            with get_openai_callback() as cb:
                profile, from_cache = cached_build_profile(
                    metrics, chunks, tenant=tenant_slug
                )
            set_llm_usage(
                settings.openai_model,
                cb.prompt_tokens,
                cb.completion_tokens,
                prompt_version=prompt_version(),
            )
            current.set_attribute("llm.cache_hit", from_cache)
    return profile, chunks, from_cache


def prompt_version() -> str:
    """Short hash of the system prompt: recorded with every profile."""
    try:
        return hashlib.sha256(SYSTEM_PROMPT_PATH.read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


# ── Persistence ──────────────────────────────────────────────────────────────


async def ingest_statement_file(
    *,
    tenant_id: uuid.UUID,
    tenant_slug: str,
    customer_id: uuid.UUID,
    uploaded_by: uuid.UUID | None,
    filename: str,
    content: bytes,
    uploaded_by_email: str | None = None,
) -> uuid.UUID:
    """Parse, sanitize, scan, categorize, compute, and store one statement."""
    from app.platform.tracing import (
        ATTR_STATEMENT_ID,
        ATTR_TENANT,
        ATTR_TENANT_ID,
        span,
    )

    with span(
        "statement.ingest",
        **{ATTR_TENANT: tenant_slug, ATTR_TENANT_ID: str(tenant_id)},
    ) as ingest_span:
        categorized, metrics, source_type, scan = await asyncio.to_thread(
            _analyze_sync, filename, content, tenant_slug
        )
        statement_id = await _persist_statement(
            tenant_id,
            customer_id,
            uploaded_by,
            filename,
            source_type,
            categorized,
            metrics,
            guardrail_flags=scan.as_dict() if scan.flagged_rows else {},
        )
        ingest_span.set_attribute(ATTR_STATEMENT_ID, str(statement_id))
        if scan.flagged_rows:
            from app.graph import audit

            await audit.record(
                tenant_id,
                statement_id=statement_id,
                node="guardrail.statement_scan",
                event="neutralised",
                actor=uploaded_by_email or "system",
                payload=scan.as_dict(),
            )
        return statement_id


async def _persist_statement(
    tenant_id: uuid.UUID,
    customer_id: uuid.UUID,
    uploaded_by: uuid.UUID | None,
    filename: str,
    source_type: SourceType,
    categorized: pd.DataFrame,
    metrics: FinancialMetrics,
    guardrail_flags: dict | None = None,
) -> uuid.UUID:
    async with tenant_session(tenant_id) as session:
        customer = await session.get(Customer, customer_id)
        if customer is None:
            raise NotFound("customer not found")

        statement = Statement(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            customer_id=customer_id,
            uploaded_by=uploaded_by,
            filename=filename[:255],
            source_type=source_type,
            period=metrics.period,
            transaction_count=metrics.transaction_count,
            status=StatementStatus.analyzed,
            guardrail_flags=guardrail_flags or {},
        )
        session.add(statement)
        await session.flush()

        rows = []
        for record in categorized.itertuples(index=False):
            txn_date = pd.to_datetime(getattr(record, "date"), errors="coerce")
            rows.append(
                Transaction(
                    tenant_id=tenant_id,
                    statement_id=statement.id,
                    txn_date=(
                        txn_date.date()
                        if pd.notna(txn_date)
                        else pd.Timestamp("1970-01-01").date()
                    ),
                    description=str(getattr(record, "description"))[:2000],
                    amount=Decimal(str(round(float(getattr(record, "amount")), 2))),
                    txn_type=str(getattr(record, "type")).strip()[:16],
                    category=str(getattr(record, "category"))[:64],
                )
            )
        session.add_all(rows)

        session.add(
            StatementMetrics(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                statement_id=statement.id,
                total_income=Decimal(str(round(metrics.total_income, 2))),
                total_expenses=Decimal(str(round(metrics.total_expenses, 2))),
                savings_rate_pct=Decimal(str(round(metrics.savings_rate_pct, 2))),
                expense_to_income_ratio=Decimal(
                    str(round(metrics.expense_to_income_ratio, 4))
                ),
                risk_band=str(
                    getattr(metrics.risk_profile, "value", metrics.risk_profile)
                ),
                health_score=int(metrics.financial_health_score),
                metrics_json=metrics.model_dump(mode="json"),
            )
        )
        logger.info(
            "Stored statement %s for customer %s (%d transactions, risk=%s)",
            statement.id,
            customer.external_ref,
            len(rows),
            metrics.risk_profile,
        )
        return statement.id


async def list_statements(tenant_id: uuid.UUID) -> list[dict]:
    async with tenant_session(tenant_id) as session:
        result = await session.execute(
            select(Statement)
            .options(selectinload(Statement.customer), selectinload(Statement.metrics))
            .order_by(Statement.created_at.desc())
        )
        return [_summary(s) for s in result.scalars().all()]


async def get_statement(tenant_id: uuid.UUID, statement_id: uuid.UUID) -> dict:
    async with tenant_session(tenant_id) as session:
        statement = await _load_statement(session, statement_id)
        result = await session.execute(
            select(Transaction)
            .where(Transaction.statement_id == statement.id)
            .order_by(Transaction.txn_date, Transaction.id)
        )
        transactions = [
            {
                "date": t.txn_date.isoformat(),
                "description": t.description,
                "amount": float(t.amount),
                "type": t.txn_type,
                "category": t.category,
            }
            for t in result.scalars().all()
        ]
        detail = _summary(statement)
        detail["metrics"] = statement.metrics.metrics_json if statement.metrics else {}
        detail["transactions"] = transactions
        detail["declared_monthly_income"] = float(
            statement.customer.declared_monthly_income
        )
        return detail


async def _load_statement(session, statement_id: uuid.UUID) -> Statement:
    result = await session.execute(
        select(Statement)
        .options(selectinload(Statement.customer), selectinload(Statement.metrics))
        .where(Statement.id == statement_id)
    )
    statement = result.scalar_one_or_none()
    if statement is None:
        # Either it does not exist or RLS hid it. The caller cannot tell,
        # and must not be able to.
        raise NotFound("statement not found")
    return statement


def _summary(statement: Statement) -> dict:
    metrics = statement.metrics
    return {
        "id": statement.id,
        "customer_id": statement.customer_id,
        "customer_name": statement.customer.full_name,
        "filename": statement.filename,
        "source_type": statement.source_type.value,
        "period": statement.period,
        "transaction_count": statement.transaction_count,
        "status": statement.status.value,
        "risk_band": metrics.risk_band if metrics else None,
        "health_score": metrics.health_score if metrics else None,
        "savings_rate_pct": float(metrics.savings_rate_pct) if metrics else None,
        "guardrail_flags": statement.guardrail_flags or {},
        "created_at": statement.created_at,
    }


async def generate_profile(
    tenant_id: uuid.UUID, tenant_slug: str, statement_id: uuid.UUID
) -> dict:
    """Retrieve + narrate for a stored statement, then persist the profile."""
    async with tenant_session(tenant_id) as session:
        statement = await _load_statement(session, statement_id)
        if statement.metrics is None:
            raise NotFound("statement has no metrics")
        metrics = FinancialMetrics.model_validate(statement.metrics.metrics_json)

    profile, chunks, from_cache = await asyncio.to_thread(
        _profile_sync, metrics, tenant_slug
    )

    async with tenant_session(tenant_id) as session:
        statement = await _load_statement(session, statement_id)
        row = Profile(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            statement_id=statement.id,
            model=settings.openai_model,
            prompt_version=prompt_version(),
            retrieved_sources=sorted({c["source"] for c in chunks}),
            profile_json=profile.model_dump(mode="json"),
            from_cache=from_cache,
        )
        session.add(row)
        statement.status = StatementStatus.profiled
        await session.flush()
        return _profile_out(row)


async def latest_profile(tenant_id: uuid.UUID, statement_id: uuid.UUID) -> dict | None:
    async with tenant_session(tenant_id) as session:
        await _load_statement(session, statement_id)
        result = await session.execute(
            select(Profile)
            .where(Profile.statement_id == statement_id)
            .order_by(Profile.created_at.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return _profile_out(row) if row else None


def _profile_out(row: Profile) -> dict:
    return {
        "id": row.id,
        "statement_id": row.statement_id,
        "model": row.model,
        "prompt_version": row.prompt_version,
        "retrieved_sources": list(row.retrieved_sources),
        "from_cache": row.from_cache,
        "profile": row.profile_json,
        "created_at": row.created_at,
    }


async def chat_context(
    tenant_id: uuid.UUID, statement_id: uuid.UUID
) -> tuple[FinancialMetrics, pd.DataFrame]:
    """Rebuild the exact (metrics, categorized_df) the chat tools close over."""
    detail = await get_statement(tenant_id, statement_id)
    metrics = FinancialMetrics.model_validate(detail["metrics"])
    df = pd.DataFrame(detail["transactions"])
    if df.empty:
        df = pd.DataFrame(columns=["date", "description", "amount", "type", "category"])
    return metrics, df


# ── Customers ────────────────────────────────────────────────────────────────


async def list_customers(tenant_id: uuid.UUID) -> list[dict]:
    async with tenant_session(tenant_id) as session:
        counts = (
            select(Statement.customer_id, func.count().label("n"))
            .group_by(Statement.customer_id)
            .subquery()
        )
        result = await session.execute(
            select(Customer, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.customer_id == Customer.id)
            .order_by(Customer.external_ref)
        )
        return [
            {
                "id": c.id,
                "external_ref": c.external_ref,
                "full_name": c.full_name,
                "declared_monthly_income": float(c.declared_monthly_income),
                "statement_count": int(n),
            }
            for c, n in result.all()
        ]


async def create_customer(
    tenant_id: uuid.UUID, external_ref: str, full_name: str, declared: float
) -> dict:
    async with tenant_session(tenant_id) as session:
        customer = Customer(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            external_ref=external_ref,
            full_name=full_name,
            declared_monthly_income=Decimal(str(round(declared, 2))),
        )
        session.add(customer)
        await session.flush()
        return {
            "id": customer.id,
            "external_ref": customer.external_ref,
            "full_name": customer.full_name,
            "declared_monthly_income": float(customer.declared_monthly_income),
            "statement_count": 0,
        }
