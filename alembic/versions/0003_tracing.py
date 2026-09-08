"""Tracing: spans table (offline trace viewer) and the prompt version registry.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-08

`spans` is written by the exporter as the owner and read by the API role
under row-level security. `prompt_versions` is global (prompts are code).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"


def upgrade() -> None:
    op.create_table(
        "spans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(32), nullable=False),
        sa.Column("span_id", sa.String(16), nullable=False),
        sa.Column("parent_span_id", sa.String(16), nullable=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.Numeric(12, 3), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("statement_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", sa.String(32), nullable=True),
        sa.Column("model", sa.String(64), nullable=True),
        sa.Column("tokens_in", sa.Integer(), nullable=True),
        sa.Column("tokens_out", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column(
            "attributes", postgresql.JSONB(), nullable=False, server_default="{}"
        ),
    )
    op.create_index("ix_spans_trace", "spans", ["trace_id"])
    op.create_index("ix_spans_run", "spans", ["run_id"])
    op.create_index("ix_spans_statement", "spans", ["statement_id"])
    op.execute(f"GRANT SELECT ON spans TO {APP_ROLE}")
    op.execute("ALTER TABLE spans ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON spans
            USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
        """)

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prompt_name", sa.String(64), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("chars", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("uses", sa.Integer(), nullable=False, server_default="0"),
        sa.UniqueConstraint(
            "prompt_name", "version", "model", name="uq_prompt_version_model"
        ),
    )
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON prompt_versions TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON SEQUENCE prompt_versions_id_seq TO {APP_ROLE}")


def downgrade() -> None:
    op.drop_table("prompt_versions")
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON spans")
    op.drop_table("spans")
