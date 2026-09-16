"""Least privilege for the API process, and retention that reaches spans and the query log.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-16

Before this revision the API process held the owner connection for three
jobs: exporting spans, reading the daily budget, and LangGraph's checkpoint
tables. None of those needs to own anything. After it:

- `spans`: the API role may INSERT (the exporter pins each tenant before
  writing, and the existing policy's USING clause doubles as the check) and
  DELETE (the retention path, limited to the pinned tenant by the policy).
- `query_log`: the API role may DELETE, for the same retention path.
- LangGraph's checkpoint tables are created here, by the owner, using
  LangGraph's own setup, and the API role gets row access to them. The API no
  longer runs DDL at startup. A LangGraph upgrade that adds checkpoint
  migrations therefore needs a new revision that calls setup again; the API
  refuses to start and says so if the tables are behind.

The checkpoint tables still have no tenant column, so they still carry no
policy; runs are addressed through `<tenant_id>:<run_id>` thread ids.
"""

from __future__ import annotations

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"
CHECKPOINT_TABLES = (
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
)


def _owner_conninfo() -> str:
    url = op.get_bind().engine.url.render_as_string(hide_password=False)
    return url.replace("postgresql+psycopg://", "postgresql://", 1).replace(
        "postgresql+psycopg2://", "postgresql://", 1
    )


def upgrade() -> None:
    op.execute(f"GRANT INSERT, DELETE ON spans TO {APP_ROLE}")
    op.execute(f"GRANT USAGE, SELECT ON SEQUENCE spans_id_seq TO {APP_ROLE}")
    op.execute(f"GRANT DELETE ON query_log TO {APP_ROLE}")

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(_owner_conninfo()) as saver:
        saver.setup()
    for table in CHECKPOINT_TABLES:
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {APP_ROLE}")


def downgrade() -> None:
    for table in CHECKPOINT_TABLES:
        op.execute(f"REVOKE SELECT, INSERT, UPDATE, DELETE ON {table} FROM {APP_ROLE}")
    op.execute(f"REVOKE DELETE ON query_log FROM {APP_ROLE}")
    op.execute(f"REVOKE USAGE, SELECT ON SEQUENCE spans_id_seq FROM {APP_ROLE}")
    op.execute(f"REVOKE INSERT, DELETE ON spans FROM {APP_ROLE}")
