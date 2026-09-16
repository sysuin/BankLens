"""
The model gateway, the job queue, the shared cache, and the rate limit.

Gateway unit tests drive provider selection, the circuit breaker and the
budget rule without any network. Queue tests enqueue through the API and
run the worker's `process()` inline against the throwaway cluster, with
the model mocked.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.core.config import settings
from app.core.context import tenant_scope
from app.platform import gateway
from tests.conftest import ROOT, login, upload_sample
from tests.test_graph import _customer_by_ref


@pytest.fixture(autouse=True)
def _clean_gateway():
    gateway.reset_circuits()
    gateway.invalidate_budget_cache()
    yield
    gateway.reset_circuits()
    gateway.invalidate_budget_cache()


# ── provider selection ───────────────────────────────────────────────────────


def test_auto_prefers_openai_when_a_key_exists(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434/v1")
    order = [p.name for p in gateway.provider_order()]
    assert order == ["openai", "ollama"]
    assert gateway.choose("primary").spec.name == "openai"
    assert gateway.embedding_signature() == "openai:" + settings.openai_embedding_model


def test_zero_key_mode_runs_on_ollama_alone(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "openai_api_key", "")
    order = [p.name for p in gateway.provider_order()]
    assert order == ["ollama"]
    choice = gateway.choose("primary")
    assert choice.spec.name == "ollama" and choice.spec.cost_free
    assert choice.fallbacks == ()
    assert gateway.embedding_signature().startswith("ollama:")


def test_explicit_provider_wins_and_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    assert [p.name for p in gateway.provider_order()] == ["ollama", "openai"]
    monkeypatch.setattr(settings, "llm_fallback_enabled", False)
    assert [p.name for p in gateway.provider_order()] == ["ollama"]


def test_no_provider_at_all_is_an_explicit_error(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    with pytest.raises(gateway.GatewayUnavailable):
        gateway.choose()
    assert not gateway.available()


# ── circuit breaker ──────────────────────────────────────────────────────────


def test_circuit_opens_after_threshold_and_half_opens_after_cooldown(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "gateway_failure_threshold", 3)
    monkeypatch.setattr(settings, "gateway_cooldown_s", 0.05)

    for _ in range(2):
        gateway.record_failure("openai", "boom")
    assert gateway.circuit("openai").state() == "closed"
    gateway.record_failure("openai", "boom")
    assert gateway.circuit("openai").state() == "open"

    # Open circuit: the primary is skipped, the fallback serves.
    choice = gateway.choose("primary")
    assert choice.spec.name == "ollama" and choice.reason == "fallback"

    import time

    time.sleep(0.06)
    assert gateway.circuit("openai").state() == "half_open"
    choice = gateway.choose("primary")
    assert choice.spec.name == "openai" and "half-open" in choice.reason

    gateway.record_success("openai")
    assert gateway.circuit("openai").state() == "closed"


def test_every_circuit_open_is_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "gateway_failure_threshold", 1)
    monkeypatch.setattr(settings, "gateway_cooldown_s", 60)
    gateway.record_failure("openai", "x")
    gateway.record_failure("ollama", "x")
    with pytest.raises(gateway.GatewayUnavailable, match="circuit open"):
        gateway.choose()


# ── budget ───────────────────────────────────────────────────────────────────


def test_spent_budget_routes_to_the_cost_free_provider(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(gateway, "budget_for", lambda tenant: (2.5, 2.0))
    with tenant_scope("meridian"):
        choice = gateway.choose("primary")
    assert choice.spec.name == "ollama" and "budget" in choice.reason


def test_spent_budget_with_no_free_provider_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "ollama_base_url", "")
    monkeypatch.setattr(gateway, "budget_for", lambda tenant: (2.5, 2.0))
    with tenant_scope("meridian"):
        with pytest.raises(gateway.BudgetExceeded):
            gateway.choose("primary")


def test_budget_is_read_from_spans(pg_cluster, client, meridian_rm, mocked_llm):
    """Spend recorded by tracing is the spend the gateway enforces."""
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    from tests.test_graph import sse_events

    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)
    gateway.invalidate_budget_cache()
    spent, limit = gateway.budget_for("meridian")
    assert limit == settings.tenant_daily_budget_usd
    assert spent >= 0.0
    status = client.get("/platform/gateway", headers=meridian_rm).json()
    assert status["budget"]["daily_limit_usd"] == limit
    assert status["would_use"]["provider"] in ("openai", "ollama")
    assert {p["name"] for p in status["providers"]} >= {"ollama"}


# ── job queue ────────────────────────────────────────────────────────────────


def _run(coro):
    """Run on a fresh loop with fresh engines (the worker's real situation)."""
    from app.db import session as db_session
    from app.graph import builder

    db_session.reset_engines()
    builder.reset_checkpointer()
    try:
        return asyncio.run(coro)
    finally:
        db_session.reset_engines()
        builder.reset_checkpointer()


def test_jobs_are_queued_processed_and_costed(
    pg_cluster, client, meridian_rm, monkeypatch
):
    # The worker must not need the owner role: remove the owner URL, as in
    # production, before the worker runs.
    monkeypatch.setattr(
        settings,
        "database_admin_url",
        "",
    )
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    with (ROOT / "data" / "sample_1_high_saver.csv").open("rb") as fh:
        created = client.post(
            "/jobs",
            headers=meridian_rm,
            data={"customer_id": customer_id, "kind": "ingest"},
            files={"file": ("bulk_001.csv", fh, "text/csv")},
        )
    assert created.status_code == 201, created.text
    job = created.json()
    assert job["status"] == "queued" and job["attempts"] == 0

    from app import worker
    from app.db import session as db_session

    async def work():
        claimed = await worker.claim_job()
        assert claimed is not None and claimed.id == uuid.UUID(job["id"])
        assert claimed.status.value == "running" and claimed.worker
        await worker.process(claimed)
        assert await worker.claim_job() is None  # nothing left
        await db_session.dispose_engines()

    _run(work())

    finished = client.get(f"/jobs/{job['id']}", headers=meridian_rm).json()
    assert finished["status"] == "done", finished["error"]
    assert finished["statement_id"] and finished["duration_ms"] > 0
    assert finished["cost_usd"] == 0.0  # ingest never calls a model
    assert finished["attempts"] == 1
    assert (
        client.get(
            f"/statements/{finished['statement_id']}", headers=meridian_rm
        ).status_code
        == 200
    )

    summary = client.get("/jobs/summary", headers=meridian_rm).json()
    assert summary["by_status"]["done"]["count"] >= 1


def test_ingest_and_run_job_reaches_the_review_queue(
    pg_cluster, client, meridian_rm, mocked_llm
):
    # M-1003 over-declares: the graph pauses, so the job ends awaiting_review.
    customer_id = _customer_by_ref(client, meridian_rm, "M-1003")
    with (ROOT / "data" / "sample_3_cashflow_stressed.csv").open("rb") as fh:
        job = client.post(
            "/jobs",
            headers=meridian_rm,
            data={"customer_id": customer_id, "kind": "ingest_and_run"},
            files={"file": ("bulk_review.csv", fh, "text/csv")},
        ).json()

    from app import worker
    from app.db import session as db_session

    async def work():
        claimed = await worker.claim_job()
        await worker.process(claimed)
        await db_session.dispose_engines()

    _run(work())
    done = client.get(f"/jobs/{job['id']}", headers=meridian_rm).json()
    assert done["status"] == "awaiting_review" and done["run_id"]
    reviewer = login(client, "meridian", "reviewer")
    queue = client.get("/reviews", headers=reviewer).json()
    assert any(d["run_id"] == done["run_id"] for d in queue)


def test_jobs_are_tenant_isolated_and_rm_only(
    client, meridian_rm, harbor_rm, meridian_reviewer
):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    with (ROOT / "data" / "sample_statement.csv").open("rb") as fh:
        job = client.post(
            "/jobs",
            headers=meridian_rm,
            data={"customer_id": customer_id},
            files={"file": ("x.csv", fh, "text/csv")},
        ).json()
    assert client.get(f"/jobs/{job['id']}", headers=harbor_rm).status_code == 404
    assert job["id"] not in {
        j["id"] for j in client.get("/jobs", headers=harbor_rm).json()
    }
    with (ROOT / "data" / "sample_statement.csv").open("rb") as fh:
        forbidden = client.post(
            "/jobs",
            headers=meridian_reviewer,
            data={"customer_id": customer_id},
            files={"file": ("x.csv", fh, "text/csv")},
        )
    assert forbidden.status_code == 403


def test_worker_reclaims_an_expired_lease(pg_cluster, client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    with (ROOT / "data" / "sample_statement.csv").open("rb") as fh:
        job = client.post(
            "/jobs",
            headers=meridian_rm,
            data={"customer_id": customer_id},
            files={"file": ("lease.csv", fh, "text/csv")},
        ).json()

    from datetime import datetime, timedelta, timezone

    from app import worker
    from app.db import session as db_session
    from app.db.models import Job

    async def crash_then_reclaim():
        # Earlier tests may have left jobs queued; work through them first.
        first = await worker.claim_job()
        while first is not None and first.id != uuid.UUID(job["id"]):
            await worker.process(first)
            first = await worker.claim_job()
        assert first is not None
        # Simulate a dead worker: lease already expired, job still "running".
        async with db_session.admin_session() as session:
            row = await session.get(Job, first.id)
            row.leased_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        second = await worker.claim_job()
        assert second is not None and second.id == first.id and second.attempts == 2
        await worker.process(second)
        await db_session.dispose_engines()

    _run(crash_then_reclaim())
    assert (
        client.get(f"/jobs/{job['id']}", headers=meridian_rm).json()["status"] == "done"
    )


# ── shared cache ─────────────────────────────────────────────────────────────


def test_postgres_profile_cache_hits_on_the_second_call(pg_cluster, monkeypatch):
    from app.pipeline import cache
    from app.pipeline.agent import CustomerProfile
    from app.pipeline.analyzer import compute_metrics
    from app.pipeline.categorizer import categorize_dataframe
    from app.pipeline.sanitizer import sanitize_dataframe
    import pandas as pd

    monkeypatch.setattr(settings, "profile_cache_enabled", True)
    monkeypatch.setattr(settings, "profile_cache_backend", "postgres")
    df = categorize_dataframe(
        sanitize_dataframe(pd.read_csv(ROOT / "data" / "sample_statement.csv"))
    )
    metrics = compute_metrics(df)
    chunks = [{"source": "fixed_deposit.md", "content": f"unique {uuid.uuid4()}"}]

    from tests.test_graph import _profile

    calls = []

    def fake_build(metrics, chunks, tenant=None):
        calls.append(1)
        names = (
            ("Term Deposit", "High-Yield Savings Account")
            if tenant == "harbor"
            else ("Fixed Deposit", "Savings Account")
        )
        with tenant_scope(tenant or "meridian"):
            return CustomerProfile.model_validate(_profile(*names))

    monkeypatch.setattr(cache, "build_profile", fake_build)
    _, hit1 = cache.cached_build_profile(metrics, chunks, tenant="meridian")
    _, hit2 = cache.cached_build_profile(metrics, chunks, tenant="meridian")
    assert (hit1, hit2) == (False, True) and len(calls) == 1
    # Another bank never sees Meridian's cached narrative.
    _, hit3 = cache.cached_build_profile(metrics, chunks, tenant="harbor")
    assert hit3 is False and len(calls) == 2


# ── rate limit ───────────────────────────────────────────────────────────────


def test_rate_limit_returns_429_with_retry_after(client, meridian_rm, monkeypatch):
    from app.api import deps

    deps.reset_rate_limits()
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    codes = [
        client.get("/customers", headers=meridian_rm).status_code for _ in range(4)
    ]
    assert codes == [200, 200, 200, 429]
    last = client.get("/customers", headers=meridian_rm)
    assert last.headers.get("retry-after") == "60"
    deps.reset_rate_limits()
