"""Decision graph: runs, decisions (review queue), audit_events + RLS.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-08

LangGraph's own checkpoint tables (checkpoints, checkpoint_blobs,
checkpoint_writes, checkpoint_migrations) are created by the checkpointer's
setup() as the owner and are NOT tenant-scoped: they are addressed only by
thread id, which is a run id that lives in `runs` and is therefore reachable
only through an RLS-protected row. Recorded as a known limitation.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"
NEW_TABLES = ("runs", "decisions", "audit_events")


def _tenant_col():
    return sa.Column(
        "tenant_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )


def _ts(name: str, **kw):
    return sa.Column(
        name,
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
        **kw,
    )


def upgrade() -> None:
    run_status = postgresql.ENUM(
        "running",
        "awaiting_review",
        "completed",
        "failed",
        "rejected",
        name="run_status",
        create_type=False,
    )
    decision_status = postgresql.ENUM(
        "pending",
        "approved",
        "rejected",
        "auto_cleared",
        name="decision_status",
        create_type=False,
    )
    for enum_type in (run_status, decision_status):
        enum_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _tenant_col(),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "started_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", run_status, nullable=False),
        sa.Column("current_node", sa.String(64), nullable=True),
        sa.Column(
            "profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
    )
    op.create_index("ix_runs_statement", "runs", ["statement_id"])

    op.create_table(
        "decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _tenant_col(),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("declared_monthly_income", sa.Numeric(14, 2), nullable=False),
        sa.Column("observed_monthly_income", sa.Numeric(14, 2), nullable=False),
        sa.Column("discrepancy_pct", sa.Numeric(8, 2), nullable=False),
        sa.Column("threshold_pct", sa.Numeric(8, 2), nullable=False),
        sa.Column("status", decision_status, nullable=False),
        sa.Column(
            "requested_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "decided_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        _ts("created_at"),
    )
    op.create_index("ix_decisions_tenant_status", "decisions", ["tenant_id", "status"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        _tenant_col(),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("event", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(320), nullable=False),
        sa.Column("inputs_hash", sa.String(64), nullable=True),
        sa.Column("model", sa.String(64), nullable=True),
        sa.Column("prompt_version", sa.String(64), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        _ts("created_at"),
    )
    op.create_index("ix_audit_run", "audit_events", ["run_id"])
    op.create_index("ix_audit_statement", "audit_events", ["statement_id"])

    for table in NEW_TABLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON SEQUENCE audit_events_id_seq TO {APP_ROLE}")

    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
                WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
            """)
    # The trail is append-only for the API role: no UPDATE or DELETE.
    op.execute(f"REVOKE UPDATE, DELETE ON audit_events FROM {APP_ROLE}")


def downgrade() -> None:
    for table in NEW_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
    for table in ("audit_events", "decisions", "runs"):
        op.drop_table(table)
    for enum_name in ("decision_status", "run_status"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
