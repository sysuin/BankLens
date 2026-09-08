"""Platform schema: tenants, users, customers, statements, metrics, profiles + RLS.

Revision ID: 0001
Create Date: 2026-09-08

Row-level security is created here, next to the tables, so the schema and
its isolation guarantee are versioned together. The API role
(`banklens_app`) is granted table access and is subject to the policies; the
owner (whoever runs this migration) is not.

The isolation test in tests/test_tenancy.py reads tenant B's rows from a
tenant A session and expects nothing back. Comment out the policy block below
and that test fails — that is the proof the guard is real.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"

TENANT_TABLES = (
    "users",
    "customers",
    "statements",
    "transactions",
    "statement_metrics",
    "profiles",
)


def upgrade() -> None:
    user_role = postgresql.ENUM("rm", "reviewer", name="user_role", create_type=False)
    statement_source = postgresql.ENUM(
        "csv", "pdf", name="statement_source", create_type=False
    )
    statement_status = postgresql.ENUM(
        "analyzed", "profiled", name="statement_status", create_type=False
    )
    for enum_type in (user_role, statement_source, statement_status):
        enum_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "tenants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("password_hash", sa.String(200), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
    )

    op.create_table(
        "customers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("external_ref", sa.String(64), nullable=False),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column("declared_monthly_income", sa.Numeric(14, 2), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "tenant_id", "external_ref", name="uq_customers_tenant_ref"
        ),
    )

    op.create_table(
        "statements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "customer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("customers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("source_type", statement_source, nullable=False),
        sa.Column("period", sa.String(64), nullable=False),
        sa.Column("transaction_count", sa.Integer(), nullable=False),
        sa.Column("status", statement_status, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_statements_tenant_customer", "statements", ["tenant_id", "customer_id"]
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("txn_date", sa.Date(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("txn_type", sa.String(16), nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
    )
    op.create_index("ix_transactions_statement", "transactions", ["statement_id"])

    op.create_table(
        "statement_metrics",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("total_income", sa.Numeric(14, 2), nullable=False),
        sa.Column("total_expenses", sa.Numeric(14, 2), nullable=False),
        sa.Column("savings_rate_pct", sa.Numeric(8, 2), nullable=False),
        sa.Column("expense_to_income_ratio", sa.Numeric(8, 4), nullable=False),
        sa.Column("risk_band", sa.String(16), nullable=False),
        sa.Column("health_score", sa.Integer(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    op.create_table(
        "profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "tenant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("tenants.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "statement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("statements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("retrieved_sources", postgresql.JSONB(), nullable=False),
        sa.Column("profile_json", postgresql.JSONB(), nullable=False),
        sa.Column(
            "from_cache", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_profiles_statement", "profiles", ["statement_id"])

    # ── Grants for the API role ──────────────────────────────────────────────
    op.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} LOGIN;
            END IF;
        END
        $$;
        """)
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT ON tenants TO {APP_ROLE}")
    for table in TENANT_TABLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON SEQUENCE transactions_id_seq TO {APP_ROLE}")

    # ── Row-level security: the tenant filter ────────────────────────────────
    # `app.tenant_id` is SET LOCAL by app.db.session.tenant_session(). With no
    # setting the comparison is against NULL and every row is invisible.
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"""
            CREATE POLICY tenant_isolation ON {table}
                USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
                WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
            """)


def downgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
    for table in (
        "profiles",
        "statement_metrics",
        "transactions",
        "statements",
        "customers",
        "users",
        "tenants",
    ):
        op.drop_table(table)
    for enum_name in ("statement_status", "statement_source", "user_role"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
