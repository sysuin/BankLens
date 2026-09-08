"""
The decision graph, end to end through the API, with the model mocked.

What is real: the graph, LangGraph's Postgres checkpointer, the interrupt,
the resume, the review queue, roles, tenancy, and the audit trail. What is
mocked: retrieval (needs embeddings) and narration (needs the model).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.graph.nodes import discrepancy, months_in_period
from tests.conftest import ROOT, login, upload_sample

FAKE_CHUNKS = [
    {"source": "fixed_deposit.md", "content": "Fixed Deposit ..."},
    {"source": "savings_account.md", "content": "Savings Account ..."},
]


def _profile(primary="Fixed Deposit", secondary="Savings Account") -> dict:
    return {
        "financial_persona": "Disciplined Saver",
        "income_stability_analysis": "Salary credits arrive monthly without interruption.",
        "spending_pattern_breakdown": "Essential outlays dominate; discretionary spend is modest.",
        "credit_risk_assessment": "Surplus cashflow supports the computed rating.",
        "primary_product": primary,
        "primary_reason": "Idle surplus cash can be locked at a guaranteed rate.",
        "secondary_product": secondary,
        "secondary_reason": "A high-yield account keeps the buffer liquid.",
        "rm_hook_points": ["Observation", "Value proposition", "Call to action"],
        "financial_health_score": 82,
        "risk_profile": "Low",
        "retrieved_sources": ["fixed_deposit.md", "savings_account.md"],
    }


def _fake_retrieve(metrics, tenant):
    return FAKE_CHUNKS


def _fake_narrate_factory(primary="Fixed Deposit", secondary="Savings Account"):
    def _fake_narrate(metrics, chunks, tenant):
        return _profile(primary, secondary), False, 1200, 300

    return _fake_narrate


@pytest.fixture()
def mocked_llm():
    with (
        patch("app.graph.nodes._retrieve_sync", side_effect=_fake_retrieve),
        patch("app.graph.nodes._narrate_sync", side_effect=_fake_narrate_factory()),
    ):
        yield


def sse_events(
    client, method: str, url: str, headers: dict, json_body=None
) -> list[dict]:
    """Collect SSE events into [{"event": ..., ...body}]."""
    with client.stream(method, url, headers=headers, json=json_body) as response:
        assert response.status_code == 200, response.text
        raw = "".join(response.iter_text())
    events = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        kind, data = None, {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append({"event": kind, **data})
    return events


def _customer_by_ref(client, headers, ref: str) -> str:
    return next(
        c["id"]
        for c in client.get("/customers", headers=headers).json()
        if c["external_ref"] == ref
    )


def _nodes(events: list[dict]) -> list[str]:
    return [e["node"] for e in events if e["event"] == "node"]


# ── unit ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "period,months",
    [
        ("2024-03", 1),
        ("2024-01 to 2024-12", 12),
        ("2023-11 to 2024-02", 4),
        ("junk", 1),
    ],
)
def test_months_in_period(period, months):
    assert months_in_period(period) == months


def test_discrepancy_is_relative_to_declared():
    assert discrepancy(100.0, 120.0) == pytest.approx(20.0)
    assert discrepancy(100.0, 50.0) == pytest.approx(50.0)
    assert discrepancy(0.0, 50.0) == 100.0
    assert discrepancy(0.0, 0.0) == 0.0


# ── straight-through run ─────────────────────────────────────────────────────


def test_run_completes_when_income_matches(client, meridian_rm, mocked_llm):
    # M-1001 declared 200,000; sample_1 observes 2,400,000 / 12 = 200,000.
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)

    events = sse_events(
        client, "POST", f"/statements/{statement['id']}/run", meridian_rm
    )
    assert events[0]["event"] == "run" and events[-1]["event"] == "done"
    assert _nodes(events) == [
        "load_context",
        "verify_income",
        "retrieve",
        "narrate",
        "guardrails",
        "finalize",
    ]
    verify = next(e for e in events if e.get("node") == "verify_income")
    assert verify["data"]["review_required"] is False
    assert verify["data"]["discrepancy_pct"] == pytest.approx(0.0)
    assert "interrupt" not in {e["event"] for e in events}

    runs = client.get(f"/statements/{statement['id']}/runs", headers=meridian_rm).json()
    assert runs[0]["status"] == "completed" and runs[0]["profile_id"]
    profile = client.get(
        f"/statements/{statement['id']}/profile", headers=meridian_rm
    ).json()
    assert profile["profile"]["primary_product"] == "Fixed Deposit"
    assert (
        client.get(f"/statements/{statement['id']}", headers=meridian_rm).json()[
            "status"
        ]
        == "profiled"
    )

    audit = client.get(
        f"/statements/{statement['id']}/audit", headers=meridian_rm
    ).json()
    assert [a["node"] for a in audit] == [
        "start",
        "load_context",
        "verify_income",
        "retrieve",
        "narrate",
        "guardrails",
        "finalize",
    ]
    narrate = next(a for a in audit if a["node"] == "narrate")
    assert narrate["model"] and len(narrate["prompt_version"]) == 12
    assert (narrate["tokens_in"], narrate["tokens_out"]) == (1200, 300)
    assert narrate["inputs_hash"] and narrate["duration_ms"] is not None
    assert (
        next(a for a in audit if a["node"] == "verify_income")["event"]
        == "auto_cleared"
    )
    assert all(a["actor"] for a in audit)


# ── interrupt, resume in a new process ───────────────────────────────────────


def test_run_pauses_for_review_and_resumes_from_checkpoint(pg_cluster, mocked_llm):
    from fastapi.testclient import TestClient

    from app.api.app import create_app

    # Process 1: the RM starts the run; it pauses.
    with TestClient(create_app()) as client:
        rm = login(client, "meridian", "rm")
        # M-1003 declared 120,000; sample_3 observes 600,000 / 12 = 50,000 -> 58%.
        customer_id = _customer_by_ref(client, rm, "M-1003")
        statement = upload_sample(
            client, rm, customer_id, ROOT / "data" / "sample_3_cashflow_stressed.csv"
        )
        events = sse_events(client, "POST", f"/statements/{statement['id']}/run", rm)
        kinds = [e["event"] for e in events]
        assert "interrupt" in kinds and kinds[-1] == "done"
        assert _nodes(events) == ["load_context", "verify_income"]
        interrupt = next(e for e in events if e["event"] == "interrupt")
        assert interrupt["data"]["kind"] == "income_verification"
        assert interrupt["data"]["discrepancy_pct"] == pytest.approx(58.33, abs=0.01)

        run = client.get(f"/statements/{statement['id']}/runs", headers=rm).json()[0]
        assert run["status"] == "awaiting_review"
        assert run["pending_decision_id"]
        decision_id = run["pending_decision_id"]

        # The RM may not decide; the Harbor reviewer cannot even see it.
        assert (
            client.post(
                f"/reviews/{decision_id}", headers=rm, json={"action": "approve"}
            ).status_code
            == 403
        )
        harbor_reviewer = login(client, "harbor", "reviewer")
        assert (
            client.get(f"/reviews/{decision_id}", headers=harbor_reviewer).status_code
            == 404
        )
        assert (
            client.post(
                f"/reviews/{decision_id}",
                headers=harbor_reviewer,
                json={"action": "approve"},
            ).status_code
            == 404
        )
        assert decision_id not in {
            d["id"] for d in client.get("/reviews", headers=harbor_reviewer).json()
        }

        reviewer = login(client, "meridian", "reviewer")
        queue = client.get("/reviews", headers=reviewer).json()
        pending = next(d for d in queue if d["id"] == decision_id)
        assert pending["status"] == "pending"
        assert pending["customer_ref"] == "M-1003"
        assert pending["declared_monthly_income"] == 120000.0
        assert pending["observed_monthly_income"] == pytest.approx(50000.0)

    # Process 2: a fresh app, fresh checkpointer pool. Only Postgres remembers the run.
    with TestClient(create_app()) as client:
        reviewer = login(client, "meridian", "reviewer")
        events = sse_events(
            client,
            "POST",
            f"/reviews/{decision_id}",
            reviewer,
            {"action": "approve", "note": "payslips received, income confirmed"},
        )
        assert _nodes(events) == [
            "await_review",
            "retrieve",
            "narrate",
            "guardrails",
            "finalize",
        ]
        resumed = next(e for e in events if e.get("node") == "await_review")
        assert resumed["data"]["review_decision"] == "approved"
        assert events[-1]["event"] == "done"

        run = client.get(
            f"/statements/{statement['id']}/runs", headers=reviewer
        ).json()[0]
        assert run["status"] == "completed" and run["profile_id"]
        decided = client.get(f"/reviews/{decision_id}", headers=reviewer).json()
        assert decided["status"] == "approved" and decided["decided_at"]
        assert decided["note"] == "payslips received, income confirmed"
        assert (
            client.get(
                f"/statements/{statement['id']}/profile", headers=reviewer
            ).status_code
            == 200
        )

        # Deciding twice is refused.
        assert (
            client.post(
                f"/reviews/{decision_id}", headers=reviewer, json={"action": "approve"}
            ).status_code
            == 409
        )

        audit = client.get(
            f"/statements/{statement['id']}/audit", headers=reviewer
        ).json()
        events_seen = [(a["node"], a["event"]) for a in audit]
        assert ("verify_income", "review_requested") in events_seen
        assert ("await_review", "review_approved") in events_seen
        approval = next(a for a in audit if a["event"] == "review_approved")
        assert approval["actor"] == "reviewer@meridian.example"
        assert approval["payload"]["note"] == "payslips received, income confirmed"


def test_rejected_review_ends_the_run_without_a_profile(
    client, meridian_rm, meridian_reviewer, mocked_llm
):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1003")
    statement = upload_sample(
        client,
        meridian_rm,
        customer_id,
        ROOT / "data" / "sample_3_cashflow_stressed.csv",
    )
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)
    decision_id = client.get(
        f"/statements/{statement['id']}/runs", headers=meridian_rm
    ).json()[0]["pending_decision_id"]

    events = sse_events(
        client,
        "POST",
        f"/reviews/{decision_id}",
        meridian_reviewer,
        {"action": "reject", "note": "no proof of income"},
    )
    assert _nodes(events) == ["await_review", "finalize_rejected"]
    run = client.get(f"/statements/{statement['id']}/runs", headers=meridian_rm).json()[
        0
    ]
    assert run["status"] == "rejected"
    assert "no proof of income" in run["error"]
    assert (
        client.get(
            f"/statements/{statement['id']}/profile", headers=meridian_rm
        ).status_code
        == 404
    )
    assert (
        client.get(f"/statements/{statement['id']}", headers=meridian_rm).json()[
            "status"
        ]
        == "analyzed"
    )


# ── guardrails ───────────────────────────────────────────────────────────────


def test_guardrail_blocks_unsecured_credit_for_a_deficit_customer(client, meridian_rm):
    created = client.post(
        "/customers",
        headers=meridian_rm,
        json={
            "external_ref": "M-9001",
            "full_name": "Deficit Dan",
            "declared_monthly_income": 50000,
        },
    )
    assert created.status_code == 201
    statement = upload_sample(
        client,
        meridian_rm,
        created.json()["id"],
        ROOT / "data" / "sample_3_cashflow_stressed.csv",
    )
    assert (
        client.get(f"/statements/{statement['id']}", headers=meridian_rm).json()[
            "metrics"
        ]["is_cashflow_negative"]
        is True
    )

    with (
        patch("app.graph.nodes._retrieve_sync", side_effect=_fake_retrieve),
        patch(
            "app.graph.nodes._narrate_sync",
            side_effect=_fake_narrate_factory(
                primary="Credit Card", secondary="Savings Account"
            ),
        ),
    ):
        events = sse_events(
            client, "POST", f"/statements/{statement['id']}/run", meridian_rm
        )

    assert _nodes(events) == [
        "load_context",
        "verify_income",
        "retrieve",
        "narrate",
        "guardrails",
        "finalize_blocked",
    ]
    guard = next(e for e in events if e.get("node") == "guardrails")
    assert any("unsecured credit" in v for v in guard["data"]["guardrail_violations"])
    run = client.get(f"/statements/{statement['id']}/runs", headers=meridian_rm).json()[
        0
    ]
    assert run["status"] == "failed" and run["error"].startswith("guardrail:")
    assert (
        client.get(
            f"/statements/{statement['id']}/profile", headers=meridian_rm
        ).status_code
        == 404
    )
    audit = client.get(
        f"/statements/{statement['id']}/audit", headers=meridian_rm
    ).json()
    assert next(a for a in audit if a["node"] == "guardrails")["event"] == "blocked"


# ── tenancy ──────────────────────────────────────────────────────────────────


def test_runs_and_audit_are_invisible_across_tenants(
    client, meridian_rm, harbor_rm, mocked_llm
):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)

    assert (
        client.get(f"/statements/{statement['id']}/runs", headers=harbor_rm).status_code
        == 404
    )
    assert (
        client.get(
            f"/statements/{statement['id']}/audit", headers=harbor_rm
        ).status_code
        == 404
    )
    assert (
        client.post(f"/statements/{statement['id']}/run", headers=harbor_rm).status_code
        == 404
    )


def test_reviewer_cannot_start_runs(client, meridian_reviewer):
    statements = client.get("/statements", headers=meridian_reviewer).json()
    target = (
        statements[0]["id"] if statements else "00000000-0000-0000-0000-000000000000"
    )
    assert (
        client.post(f"/statements/{target}/run", headers=meridian_reviewer).status_code
        == 403
    )
