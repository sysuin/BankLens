"""The worker claims jobs through one narrow function instead of the owner role.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-16

Claiming is the one step that must look across banks: the worker does not
know which bank's job is next. Before this revision it held the owner
connection for that, and kept using it to load and finish the job.

`claim_next_job(worker, lease_seconds)` is SECURITY DEFINER, owned by the
owner, with a fixed search_path. It locks the oldest queued or lease-expired
job with SKIP LOCKED, marks it running with a lease, and returns only the
job's id and tenant id. EXECUTE is revoked from PUBLIC and granted to the API
role. The worker then loads and finishes the job inside a tenant-pinned
session, where the jobs policy applies, so no other bank's row is ever read
or written with owner rights.
"""

from __future__ import annotations

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

APP_ROLE = "banklens_app"


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION claim_next_job(p_worker text, p_lease_seconds integer)
        RETURNS TABLE (job_id uuid, job_tenant_id uuid)
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = public, pg_temp
        AS $$
            UPDATE jobs AS j
               SET status = 'running',
                   attempts = j.attempts + 1,
                   worker = p_worker,
                   leased_until = now() + make_interval(secs => p_lease_seconds),
                   started_at = now()
             WHERE j.id = (
                    SELECT c.id FROM jobs AS c
                     WHERE c.status = 'queued'
                        OR (c.status = 'running' AND c.leased_until < now())
                     ORDER BY c.created_at
                     FOR UPDATE SKIP LOCKED
                     LIMIT 1)
            RETURNING j.id, j.tenant_id
        $$
        """)
    op.execute("REVOKE ALL ON FUNCTION claim_next_job(text, integer) FROM PUBLIC")
    op.execute(f"GRANT EXECUTE ON FUNCTION claim_next_job(text, integer) TO {APP_ROLE}")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS claim_next_job(text, integer)")
