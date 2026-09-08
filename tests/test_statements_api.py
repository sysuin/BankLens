"""
Statements over the API: upload, persistence, profile, chat.

The LLM half is mocked: `service._profile_sync` and `run_chat_turn` never
reach OpenAI here. What is real is the deterministic half (parse, sanitize,
categorize, metrics), the database, and the streaming plumbing.
"""

from __future__ import annotations

from unittest.mock import patch

from app.pipeline.agent import CustomerProfile
from tests.conftest import ROOT, upload_sample


def _profile_payload() -> dict:
    return {
        "financial_persona": "Disciplined Saver",
        "income_stability_analysis": "Salary credits arrive monthly without interruption.",
        "spending_pattern_breakdown": "Essential outlays dominate; discretionary spend is modest.",
        "credit_risk_assessment": "Surplus cashflow supports a Low risk rating.",
        "primary_product": "Fixed Deposit",
        "primary_reason": "Idle surplus cash can be locked at a guaranteed rate.",
        "secondary_product": "Savings Account",
        "secondary_reason": "A high-yield account keeps the buffer liquid.",
        "rm_hook_points": ["Observation", "Value proposition", "Call to action"],
        "financial_health_score": 82,
        "risk_profile": "Low",
        "retrieved_sources": ["fixed_deposit.md", "savings_account.md"],
    }


def _customer(client, headers) -> str:
    return client.get("/customers", headers=headers).json()[0]["id"]


def test_upload_csv_persists_metrics_and_transactions(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    summary = upload_sample(client, meridian_rm, customer_id)

    assert summary["source_type"] == "csv"
    assert summary["transaction_count"] == 120
    assert summary["risk_band"] in {"Low", "Medium", "High"}
    assert summary["status"] == "analyzed"

    detail = client.get(f"/statements/{summary['id']}", headers=meridian_rm).json()
    assert len(detail["transactions"]) == 120
    assert detail["metrics"]["transaction_count"] == 120
    assert detail["metrics"]["risk_profile"] == summary["risk_band"]
    assert detail["declared_monthly_income"] > 0
    # PII masking happened before storage: no raw 16-digit card numbers.
    import re

    joined = " ".join(t["description"] for t in detail["transactions"])
    assert not re.search(r"\b\d{16}\b", joined)
    categories = {t["category"] for t in detail["transactions"]}
    assert len(categories) > 1


def test_upload_pdf_uses_the_pdf_parser(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    summary = upload_sample(
        client,
        meridian_rm,
        customer_id,
        ROOT / "data" / "sample_3_cashflow_stressed.pdf",
    )
    assert summary["source_type"] == "pdf"
    assert summary["transaction_count"] == 108


def test_bad_uploads_are_422(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    bad_columns = client.post(
        "/statements",
        headers=meridian_rm,
        data={"customer_id": customer_id},
        files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert bad_columns.status_code == 422
    wrong_type = client.post(
        "/statements",
        headers=meridian_rm,
        data={"customer_id": customer_id},
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert wrong_type.status_code == 422


def test_list_is_newest_first_and_scoped(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    first = upload_sample(client, meridian_rm, customer_id)
    second = upload_sample(client, meridian_rm, customer_id)
    ids = [s["id"] for s in client.get("/statements", headers=meridian_rm).json()]
    assert ids.index(second["id"]) < ids.index(first["id"])


def test_profile_is_generated_and_stored_with_provenance(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    summary = upload_sample(client, meridian_rm, customer_id)

    fake_chunks = [
        {"source": "fixed_deposit.md", "content": "Fixed Deposit ..."},
        {"source": "savings_account.md", "content": "Savings Account ..."},
    ]

    def fake_profile_sync(metrics, tenant_slug):
        from app.core.context import tenant_scope

        with tenant_scope(tenant_slug):
            return (
                CustomerProfile.model_validate(_profile_payload()),
                fake_chunks,
                False,
            )

    assert (
        client.get(
            f"/statements/{summary['id']}/profile", headers=meridian_rm
        ).status_code
        == 404
    )

    with patch("app.api.service._profile_sync", side_effect=fake_profile_sync):
        created = client.post(
            f"/statements/{summary['id']}/profile", headers=meridian_rm
        )
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["profile"]["primary_product"] == "Fixed Deposit"
    assert body["retrieved_sources"] == ["fixed_deposit.md", "savings_account.md"]
    assert body["model"] and len(body["prompt_version"]) == 12
    assert body["from_cache"] is False

    latest = client.get(
        f"/statements/{summary['id']}/profile", headers=meridian_rm
    ).json()
    assert latest["id"] == body["id"]
    assert (
        client.get(f"/statements/{summary['id']}", headers=meridian_rm).json()["status"]
        == "profiled"
    )


def test_chat_streams_server_sent_events(client, meridian_rm):
    customer_id = _customer(client, meridian_rm)
    summary = upload_sample(client, meridian_rm, customer_id)

    def fake_turn(question, history, metrics, df, tenant=None):
        assert tenant == "meridian"
        assert metrics.transaction_count == 120
        assert len(df) == 120
        for piece in ("The savings ", "rate is ", f"{metrics.savings_rate_pct:.1f}%."):
            yield piece
        from langchain_core.messages import AIMessage, HumanMessage

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="The savings rate is known."))

    with patch("app.pipeline.chat.run_chat_turn", side_effect=fake_turn):
        with client.stream(
            "POST",
            f"/statements/{summary['id']}/chat",
            headers=meridian_rm,
            json={
                # A product question: numeric ones are answered from the
                # warehouse without a model (see tests/test_warehouse.py).
                "question": "which product suits this customer?",
                "history": [{"role": "user", "content": "hi"}],
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            raw = "".join(response.iter_text())

    events = [block for block in raw.split("\n\n") if block.strip()]
    kinds = [block.split("\n")[0] for block in events]
    assert kinds[:3] == ["event: token"] * 3
    assert kinds[-1] == "event: done"
    assert '"The savings rate is known."' in events[-1]
