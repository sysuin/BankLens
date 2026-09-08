"""Phase 8: shared rate-limit counters, and the API role may delete customers.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-08

`rate_limit_buckets` is one row per user per minute, incremented by an
upsert from every API process, so the per-user limit holds across
processes without Redis. It is keyed by user id, carries no tenant data
and no statement text, and is swept by the limiter itself.

Customer deletion (the retention helper) is a plain DELETE on the API role
inside a tenant-pinned transaction; the row-level policy limits it to the
caller's bank and the foreign keys cascade. Grants for that already exist
from 0001; this revision only adds the counter table.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"


def upgrade() -> None:
    op.create_table(
        "rate_limit_buckets",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bucket", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("user_id", "bucket"),
    )
    op.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON rate_limit_buckets TO {APP_ROLE}"
    )


def downgrade() -> None:
    op.drop_table("rate_limit_buckets")
