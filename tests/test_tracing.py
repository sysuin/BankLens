"""
Tracing: spans land in Postgres with tenant, run, tokens and cost, and are
readable only inside the tenant. Pricing and span conversion as units.
"""

from __future__ import annotations

import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.platform import pricing
from app.platform.tracing import ATTR_TENANT_ID, _span_row
from tests.conftest import ROOT, upload_sample
from tests.test_graph import _customer_by_ref, sse_events

# ── units ────────────────────────────────────────────────────────────────────


def test_pricing_known_and_unknown_models():
    assert pricing.cost_usd("gpt-4o", 1_000_000, 0) == 2.5
    assert pricing.cost_usd("gpt-4o-2024-08-06", 0, 1_000_000) == 10.0
    assert pricing.cost_usd("gpt-4o-mini", 328, 79) == pytest.approx(
        0.0000966, abs=1e-6
    )
    assert pricing.cost_usd("llama3.1:8b", 5000, 500) == 0.0
    assert pricing.is_priced("gpt-4o") and not pricing.is_priced("llama3.1:8b")


def test_span_row_carries_tenant_tokens_and_cost():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    tenant_id = uuid.uuid4()
    with tracer.start_as_current_span("llm.profile") as span:
        span.set_attribute(ATTR_TENANT_ID, str(tenant_id))
        span.set_attribute("llm.model", "gpt-4o")
        span.set_attribute("llm.tokens.in", 2375)
        span.set_attribute("llm.tokens.out", 379)
        span.set_attribute("llm.cost_usd", pricing.cost_usd("gpt-4o", 2375, 379))
    (finished,) = exporter.get_finished_spans()
    row = _span_row(finished)
    assert row["tenant_id"] == tenant_id
    assert row["name"] == "llm.profile" and row["parent_span_id"] is None
    assert row["tokens_in"] == 2375 and row["tokens_out"] == 379
    assert row["cost_usd"] == pytest.approx(0.009728, abs=1e-6)
    assert len(row["trace_id"]) == 32 and len(row["span_id"]) == 16


def test_spans_without_a_tenant_are_not_stored():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("t").start_as_current_span("http GET /health"):
        pass
    (finished,) = exporter.get_finished_spans()
    assert _span_row(finished) is None


# ── end to end ───────────────────────────────────────────────────────────────


def test_a_run_leaves_a_trace_with_nested_node_spans(client, meridian_rm, mocked_llm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)

    traces = client.get(
        f"/statements/{statement['id']}/traces", headers=meridian_rm
    ).json()
    # The HTTP request is the root of every trace; the work nests under it.
    assert all(t["root"].startswith("http ") for t in traces)

    run_trace = next(t for t in traces if t["run_id"])
    assert run_trace["spans"] >= 8
    assert run_trace["duration_ms"] > 0

    detail = client.get(f"/traces/{run_trace['trace_id']}", headers=meridian_rm).json()
    names = [s["name"] for s in detail["spans"]]
    assert names[0].startswith("http POST") and names[1] == "graph.run"
    for node in (
        "load_context",
        "verify_income",
        "retrieve",
        "narrate",
        "guardrails",
        "finalize",
    ):
        assert f"graph.{node}" in names
    run_span_id = detail["spans"][1]["span_id"]
    node_spans = [
        s
        for s in detail["spans"]
        if s["name"].startswith("graph.") and s["name"] != "graph.run"
    ]
    assert all(
        s["parent_span_id"] == run_span_id and s["depth"] == 2 for s in node_spans
    )
    assert detail["run_id"] and detail["statement_id"] == statement["id"]
    assert len(detail["waterfall"]) == len(detail["spans"])
    assert all(
        "banklens." not in key for s in detail["spans"] for key in s["attributes"]
    )

    ingest = next(
        t for t in traces if not t["run_id"] and t["root"].endswith("/statements")
    )
    ingest_detail = client.get(
        f"/traces/{ingest['trace_id']}", headers=meridian_rm
    ).json()
    stage_names = [s["name"] for s in ingest_detail["spans"]]
    assert "statement.ingest" in stage_names
    for stage in (
        "pipeline.parse",
        "pipeline.sanitize",
        "pipeline.categorize",
        "pipeline.metrics",
    ):
        assert stage in stage_names
    categorize = next(
        s for s in ingest_detail["spans"] if s["name"] == "pipeline.categorize"
    )
    assert categorize["attributes"]["pipeline.llm_fallback_rows"] == 0


def test_traces_are_tenant_isolated(client, meridian_rm, harbor_rm, mocked_llm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)
    trace_id = client.get(
        f"/statements/{statement['id']}/traces", headers=meridian_rm
    ).json()[0]["trace_id"]

    assert (
        client.get(
            f"/statements/{statement['id']}/traces", headers=harbor_rm
        ).status_code
        == 404
    )
    assert client.get(f"/traces/{trace_id}", headers=harbor_rm).status_code == 404
    assert client.get("/traces/not-a-trace", headers=harbor_rm).status_code == 404


def test_prompt_registry_records_the_version_that_ran(client, meridian_rm, mocked_llm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)

    versions = client.get("/platform/prompts", headers=meridian_rm).json()
    assert versions, "narrate should register the prompt version"
    from app.api.service import prompt_version

    row = next(v for v in versions if v["version"] == prompt_version())
    assert row["prompt_name"] == "system_prompt" and row["uses"] >= 1
    assert len(row["sha256"]) == 64 and row["chars"] > 0


def test_chat_turn_is_a_span_with_tool_children(client, meridian_rm):
    from unittest.mock import patch

    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(
        client, meridian_rm, customer_id, ROOT / "data" / "sample_statement.csv"
    )

    def fake_turn(question, history, metrics, df, tenant=None):
        from langchain_core.messages import AIMessage, HumanMessage

        from app.platform.tracing import span

        with span("tool.get_customer_metrics"):
            pass
        yield "ok"
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="ok"))

    with patch("app.pipeline.chat.run_chat_turn", side_effect=fake_turn):
        with client.stream(
            "POST",
            f"/statements/{statement['id']}/chat",
            headers=meridian_rm,
            json={"question": "metrics?"},
        ) as response:
            "".join(response.iter_text())

    traces = client.get(
        f"/statements/{statement['id']}/traces", headers=meridian_rm
    ).json()
    chat = next(t for t in traces if t["root"].endswith("/chat"))
    detail = client.get(f"/traces/{chat['trace_id']}", headers=meridian_rm).json()
    names = [s["name"] for s in detail["spans"]]
    assert names[1:] == ["llm.chat_turn", "tool.get_customer_metrics"]
    assert [s["depth"] for s in detail["spans"]] == [0, 1, 2]
    assert detail["spans"][1]["model"] == "gpt-4o"
