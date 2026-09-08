"""Phase 8: the backlog, each item with a proof.

- checkpoints are namespaced by tenant and can be purged by run
- customer deletion cascades through the tenant tables and the checkpoints,
  and leaves an audit row with counts only
- the rate limit holds across processes when its backend is Postgres
- the golden set carries each bank's own credit policy
- the pilot report reads real timestamps and labels the assumption
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.core.config import settings
from tests.conftest import ROOT, login, run_db, upload_sample
from tests.test_graph import sse_events

SAMPLE_3 = ROOT / "data" / "sample_3_cashflow_stressed.csv"


def _paused_run(client, rm, ref: str) -> tuple[str, str]:
    """Create a customer whose declared income disagrees; start a run; it pauses."""
    customer = client.post(
        "/customers",
        headers=rm,
        json={
            "external_ref": ref,
            "full_name": "Synthetic Person",
            "declared_monthly_income": 120000,
        },
    ).json()
    statement = upload_sample(client, rm, customer["id"], SAMPLE_3)
    events = sse_events(client, "POST", f"/statements/{statement['id']}/run", rm)
    assert "interrupt" in {e["event"] for e in events}
    run_id = next(e["run_id"] for e in events if e["event"] == "run")
    return customer["id"], run_id


def _tenant_id(pg_cluster, slug: str) -> uuid.UUID:
    async def go(engine):
        async with engine.connect() as conn:
            return (
                await conn.execute(
                    text("SELECT id FROM tenants WHERE slug = :s"), {"s": slug}
                )
            ).scalar_one()

    return run_db(pg_cluster["admin_url"], go)


def _checkpoint_rows(pg_cluster, thread: str) -> int:
    async def go(engine):
        async with engine.connect() as conn:
            total = 0
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                total += (
                    await conn.execute(
                        text(f"SELECT count(*) FROM {table} WHERE thread_id = :t"),
                        {"t": thread},
                    )
                ).scalar_one()
            return total

    return run_db(pg_cluster["admin_url"], go)


# ── checkpoints and retention ────────────────────────────────────────────────


def test_checkpoints_are_namespaced_by_tenant(
    client, meridian_rm, pg_cluster, mocked_llm
):
    from app.graph.builder import thread_id

    _, run_id = _paused_run(client, meridian_rm, "M-P8-NS")
    meridian = _tenant_id(pg_cluster, "meridian")
    harbor = _tenant_id(pg_cluster, "harbor")
    assert _checkpoint_rows(pg_cluster, thread_id(meridian, run_id)) > 0
    # The same run id under the other bank's namespace addresses nothing.
    assert _checkpoint_rows(pg_cluster, thread_id(harbor, run_id)) == 0
    assert _checkpoint_rows(pg_cluster, run_id) == 0


def test_deleting_a_customer_removes_everything_and_leaves_a_count(
    client, meridian_rm, meridian_reviewer, pg_cluster, mocked_llm
):
    from app.graph.builder import thread_id

    customer_id, run_id = _paused_run(client, meridian_rm, "M-P8-DEL")
    meridian = _tenant_id(pg_cluster, "meridian")
    thread = thread_id(meridian, run_id)
    assert _checkpoint_rows(pg_cluster, thread) > 0

    # An RM may not delete; the other bank's reviewer cannot see the customer.
    assert (
        client.delete(f"/customers/{customer_id}", headers=meridian_rm).status_code
        == 403
    )
    harbor_reviewer = login(client, "harbor", "reviewer")
    assert (
        client.delete(f"/customers/{customer_id}", headers=harbor_reviewer).status_code
        == 404
    )

    response = client.delete(f"/customers/{customer_id}", headers=meridian_reviewer)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["statements"] == 1 and body["runs"] == 1
    assert body["checkpoint_rows"] > 0
    assert _checkpoint_rows(pg_cluster, thread) == 0
    assert customer_id not in {
        c["id"] for c in client.get("/customers", headers=meridian_rm).json()
    }

    async def go(engine):
        async with engine.connect() as conn:
            left = {}
            for table, column in (
                ("statements", "customer_id"),
                ("runs", "id"),
                ("decisions", "run_id"),
            ):
                value = customer_id if table == "statements" else run_id
                left[table] = (
                    await conn.execute(
                        text(f"SELECT count(*) FROM {table} WHERE {column} = :v"),
                        {"v": value},
                    )
                ).scalar_one()
            audit = (
                await conn.execute(
                    text(
                        "SELECT payload FROM audit_events "
                        "WHERE node = 'retention' AND event = 'customer_deleted' "
                        "ORDER BY created_at DESC LIMIT 1"
                    )
                )
            ).scalar_one()
            return left, audit

    left, audit = run_db(pg_cluster["admin_url"], go)
    assert left == {"statements": 0, "runs": 0, "decisions": 0}
    assert audit["customer_ref"] == "M-P8-DEL"
    assert audit["runs"] == 1 and audit["checkpoint_rows"] == body["checkpoint_rows"]
    assert "Synthetic Person" not in str(audit)


# ── rate limiting across processes ───────────────────────────────────────────


def test_postgres_limiter_is_shared_between_instances(pg_cluster):
    from app.platform.ratelimit import PostgresLimiter

    user = uuid.uuid4()

    async def go(engine):
        # Two limiter objects = two API processes. They share one counter.
        first, second = PostgresLimiter(lambda: engine), PostgresLimiter(lambda: engine)
        await first.reset()
        results = [
            await first.hit(user, 3),
            await second.hit(user, 3),
            await first.hit(user, 3),
            await second.hit(user, 3),
        ]
        other = await second.hit(uuid.uuid4(), 3)
        await first.reset()
        return results, other

    results, other = run_db(pg_cluster["app_url"], go)
    assert results[:3] == [None, None, None]
    assert isinstance(results[3], int) and 1 <= results[3] <= 60
    assert other is None


def test_api_uses_the_postgres_limiter_when_configured(
    client, meridian_rm, monkeypatch
):
    from app.api import deps

    deps.reset_rate_limits()
    monkeypatch.setattr(settings, "rate_limit_backend", "postgres")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    try:
        codes = [
            client.get("/customers", headers=meridian_rm).status_code for _ in range(4)
        ]
        assert codes == [200, 200, 200, 429]
        last = client.get("/customers", headers=meridian_rm)
        assert last.headers.get("retry-after", "").isdigit()
    finally:
        monkeypatch.setattr(settings, "rate_limit_backend", "memory")
        deps.reset_rate_limits()


# ── the golden set per bank ──────────────────────────────────────────────────


def test_golden_set_carries_each_banks_credit_policy():
    from app.pipeline.policy import forbidden_in_deficit
    from evals.dataset import build_cases

    meridian = {c.case_id: c for c in build_cases("meridian")}
    harbor = {c.case_id: c for c in build_cases("harbor")}
    assert meridian.keys() == harbor.keys()
    deficit = [cid for cid, c in meridian.items() if c.forbidden_products]
    assert deficit
    for cid in deficit:
        assert meridian[cid].forbidden_products == forbidden_in_deficit("meridian")
        assert harbor[cid].forbidden_products == forbidden_in_deficit("harbor")
        assert "auto_loan.md" in harbor[cid].forbidden_products
        assert "auto_loan.md" not in meridian[cid].forbidden_products
    for cid, case in meridian.items():
        assert case.expected_risk == harbor[cid].expected_risk


def test_graph_and_evals_share_one_policy():
    from app.graph import nodes
    from app.pipeline import policy

    assert (
        nodes.CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT
        is policy.CREDIT_PRODUCTS_FORBIDDEN_IN_DEFICIT
    )


# ── the pilot report ─────────────────────────────────────────────────────────


def test_pilot_report_reads_timestamps_and_labels_the_assumption(
    client, meridian_rm, meridian_reviewer, pg_cluster, mocked_llm
):
    from app.warehouse.pilot import pilot_summary, render

    _, run_id = _paused_run(client, meridian_rm, "M-P8-PILOT")
    decision = next(
        d
        for d in client.get("/reviews", headers=meridian_reviewer).json()
        if d["run_id"] == run_id
    )
    events = sse_events(
        client,
        "POST",
        f"/reviews/{decision['id']}",
        meridian_reviewer,
        {"action": "approve"},
    )
    assert events[-1]["event"] == "done"

    async def go(engine):
        return await pilot_summary("meridian", engine)

    summary = run_db(pg_cluster["app_url"], go)
    assert summary["runs_completed"] >= 1
    assert summary["reviews_decided"] >= 1
    assert summary["seconds_to_profile_p50"] is not None
    assert summary["seconds_to_profile_p50"] >= 0
    assert summary["reviewer_wait_seconds_p50"] is not None
    assert summary["manual_minutes_assumed"] == 20.0
    report = render(summary)
    assert "ASSUMPTION" in report and "if the assumption holds" in report

    with pytest.raises(LookupError):
        run_db(pg_cluster["app_url"], lambda e: pilot_summary("nobody", e))
