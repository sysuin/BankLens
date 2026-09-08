"""Warehouse: tenant-filtered views, a read-only chat role, and the query log.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-08

The chat's numeric questions never touch base tables. They run as
`banklens_chat`, a role that can SELECT only from views whose definitions
carry the tenant predicate (`tenant_id = current_setting('app.tenant_id')`).
The views run as their owner, so the chat role needs no privilege on the
underlying tables and has none: a crafted question that reaches SQL cannot
see another bank, and cannot see anything the views do not expose.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"
CHAT_ROLE = "banklens_chat"
TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"

VIEWS = {
    "v_customers": f"""
        SELECT c.id AS customer_id, c.external_ref, c.full_name,
               c.declared_monthly_income,
               (SELECT COUNT(*) FROM statements s WHERE s.customer_id = c.id) AS statement_count
        FROM customers c
        WHERE c.tenant_id = {TENANT}
    """,
    "v_statement_metrics": f"""
        SELECT s.id AS statement_id, s.customer_id, c.external_ref, c.full_name,
               s.period, s.transaction_count, s.status::text AS status,
               m.total_income, m.total_expenses, m.savings_rate_pct,
               m.expense_to_income_ratio, m.risk_band, m.health_score,
               (m.metrics_json->>'essential_expenses')::numeric AS essential_expenses,
               (m.metrics_json->>'discretionary_expenses')::numeric AS discretionary_expenses,
               (m.metrics_json->>'is_cashflow_negative')::boolean AS is_cashflow_negative,
               c.declared_monthly_income,
               s.created_at
        FROM statements s
        JOIN customers c ON c.id = s.customer_id
        LEFT JOIN statement_metrics m ON m.statement_id = s.id
        WHERE s.tenant_id = {TENANT}
    """,
    "v_transactions": f"""
        SELECT t.statement_id, t.txn_date, t.description, t.amount, t.txn_type, t.category
        FROM transactions t
        WHERE t.tenant_id = {TENANT}
    """,
    "v_decisions": f"""
        SELECT d.id AS decision_id, d.run_id, d.statement_id, c.external_ref, c.full_name,
               d.kind, d.declared_monthly_income, d.observed_monthly_income,
               d.discrepancy_pct, d.threshold_pct, d.status::text AS status,
               d.note, d.decided_at, d.created_at,
               u.email AS decided_by
        FROM decisions d
        JOIN statements s ON s.id = d.statement_id
        JOIN customers c ON c.id = s.customer_id
        LEFT JOIN users u ON u.id = d.decided_by
        WHERE d.tenant_id = {TENANT}
    """,
}


def upgrade() -> None:
    op.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{CHAT_ROLE}') THEN
                CREATE ROLE {CHAT_ROLE} LOGIN NOINHERIT;
            END IF;
        END
        $$;
        """)
    op.execute(f"GRANT USAGE ON SCHEMA public TO {CHAT_ROLE}")
    for name, body in VIEWS.items():
        op.execute(f"CREATE OR REPLACE VIEW {name} AS {body}")
        op.execute(f"GRANT SELECT ON {name} TO {CHAT_ROLE}")
        op.execute(f"GRANT SELECT ON {name} TO {APP_ROLE}")
    # Belt and braces: the chat role must hold nothing on base tables.
    for table in (
        "tenants",
        "users",
        "customers",
        "statements",
        "transactions",
        "statement_metrics",
        "profiles",
        "runs",
        "decisions",
        "audit_events",
        "spans",
        "jobs",
        "profile_cache",
        "prompt_versions",
    ):
        op.execute(f"REVOKE ALL ON {table} FROM {CHAT_ROLE}")

    op.create_table(
        "query_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("template", sa.String(64), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("actor", sa.String(320), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("statement_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("question_hash", sa.String(16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_query_log_tenant_created", "query_log", ["tenant_id", "created_at"]
    )
    op.execute(f"GRANT SELECT, INSERT ON query_log TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON SEQUENCE query_log_id_seq TO {APP_ROLE}")
    op.execute("ALTER TABLE query_log ENABLE ROW LEVEL SECURITY")
    op.execute(f"""
        CREATE POLICY tenant_isolation ON query_log
            USING (tenant_id = {TENANT}) WITH CHECK (tenant_id = {TENANT})
        """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON query_log")
    op.drop_table("query_log")
    for name in VIEWS:
        op.execute(f"DROP VIEW IF EXISTS {name}")
