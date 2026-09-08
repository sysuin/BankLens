"""
The warehouse: semantic layer validation, the chat role's reach, the router,
templates through the chat, role denial, and the query log.

No model is called on the template path; that is the point.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.warehouse import semantic
from app.warehouse.router import route
from tests.conftest import ROOT, login, run_db, upload_sample
from tests.test_graph import _customer_by_ref, sse_events

# ── semantic layer ───────────────────────────────────────────────────────────


def test_layer_loads_and_every_template_is_allow_listed():
    layer = semantic.load_layer()
    assert layer.relations >= {"v_statement_metrics", "v_transactions", "v_decisions"}
    assert {"savings_rate", "pending_reviews", "category_spend"} <= set(layer.templates)
    assert all(t.allows("reviewer") for t in layer.templates.values())
    rm_only = {t.name for t in semantic.templates_for("rm")}
    assert "pending_reviews" not in rm_only and "savings_rate" in rm_only


def test_unsafe_template_fails_at_load(tmp_path: Path):
    bad = tmp_path / "layer.yaml"
    bad.write_text(
        """
relations: [v_transactions]
templates:
  leak:
    roles: [rm]
    triggers: [leak]
    params: []
    sql: SELECT * FROM statements LIMIT 5
""",
        encoding="utf-8",
    )
    semantic.load_layer.cache_clear()
    try:
        with pytest.raises(semantic.SemanticLayerError, match="relation not allowed"):
            semantic.load_layer(bad)
    finally:
        semantic.load_layer.cache_clear()


def test_template_must_declare_its_binds(tmp_path: Path):
    bad = tmp_path / "layer.yaml"
    bad.write_text(
        """
relations: [v_transactions]
templates:
  t:
    roles: [rm]
    triggers: [t]
    params: [statement_id]
    sql: SELECT amount FROM v_transactions WHERE statement_id = :statement_id AND category = :category LIMIT 5
""",
        encoding="utf-8",
    )
    semantic.load_layer.cache_clear()
    try:
        with pytest.raises(semantic.SemanticLayerError, match="does not declare"):
            semantic.load_layer(bad)
    finally:
        semantic.load_layer.cache_clear()


# ── the chat role's reach ────────────────────────────────────────────────────


def test_chat_role_reads_views_only(pg_cluster):
    from sqlalchemy import text

    from app.db.local import local_chat_url

    ids = pg_cluster["ids"]["tenants"]
    chat_url = local_chat_url(pg_cluster["pgdata"])

    async def probe(engine):
        async with engine.connect() as conn:
            async with conn.begin():
                await conn.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(ids["meridian"])},
                )
                customers = (
                    await conn.execute(text("SELECT external_ref FROM v_customers"))
                ).fetchall()
            try:
                async with conn.begin():
                    await conn.execute(text("SELECT count(*) FROM statements"))
                denied = False
            except Exception as exc:  # noqa: BLE001
                denied = "permission denied" in str(exc)
            return [c[0] for c in customers], denied

    refs, denied = run_db(chat_url, probe)
    assert refs and all(r.startswith("M-") for r in refs)
    assert denied, "the chat role must not read base tables"


def test_views_filter_by_tenant_and_show_nothing_unpinned(pg_cluster):
    from sqlalchemy import text

    from app.db.local import local_chat_url

    ids = pg_cluster["ids"]["tenants"]

    async def probe(engine):
        async with engine.connect() as conn:
            async with conn.begin():
                unpinned = (
                    await conn.execute(text("SELECT count(*) FROM v_customers"))
                ).scalar_one()
            async with conn.begin():
                await conn.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(ids["harbor"])},
                )
                harbor = [
                    r[0]
                    for r in (
                        await conn.execute(text("SELECT external_ref FROM v_customers"))
                    ).fetchall()
                ]
            return unpinned, harbor

    unpinned, harbor = run_db(local_chat_url(pg_cluster["pgdata"]), probe)
    assert unpinned == 0
    assert harbor and all(r.startswith("H-") for r in harbor)


# ── router ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What is the savings rate?", "savings_rate"),
        ("Why is this customer rated High risk?", "risk_band"),
        ("top 3 categories", "top_categories"),
        ("How much did they spend on food?", "category_spend"),
        (
            "Explain the declared vs observed income discrepancy.",
            "declared_vs_observed",
        ),
    ],
)
def test_router_picks_the_template(question, expected):
    routed = route(
        question, role="rm", statement_id="s1", categories=["Food", "Shopping"]
    )
    assert routed.kind == "template" and routed.template.name == expected


def test_router_extracts_limit_and_category():
    routed = route("top 3 categories", role="rm", statement_id="s1", categories=[])
    assert routed.params["limit"] == 3
    routed = route(
        "how much on shopping",
        role="rm",
        statement_id="s1",
        categories=["Food", "Shopping"],
    )
    assert routed.params["category"] == "Shopping"


def test_router_sends_product_questions_to_chat_and_denies_reviewer_templates():
    assert (
        route("Which product would you recommend?", role="rm", statement_id="s1").kind
        == "chat"
    )
    denied = route("show me the pending reviews", role="rm", statement_id="s1")
    assert denied.kind == "denied" and denied.template.name == "pending_reviews"
    allowed = route("show me the pending reviews", role="reviewer", statement_id="s1")
    assert allowed.kind == "template"


# ── through the chat ─────────────────────────────────────────────────────────


def _never_called(*args, **kwargs):
    raise AssertionError("a template question must not reach the model")


def test_numeric_question_is_answered_from_the_warehouse(client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    with patch("app.pipeline.chat.run_chat_turn", side_effect=_never_called):
        events = sse_events(
            client,
            "POST",
            f"/statements/{statement['id']}/chat",
            meridian_rm,
            {"question": "What is the savings rate?"},
        )
    assert events[0]["event"] == "template" and events[0]["template"] == "savings_rate"
    assert events[0]["rows"] == 1
    answer = events[-1]["data"]
    detail = client.get(f"/statements/{statement['id']}", headers=meridian_rm).json()
    assert f"{detail['metrics']['savings_rate_pct']:.1f}%" in answer
    assert "computed in code" in answer

    log = client.get("/warehouse/query-log", headers=meridian_rm).json()
    assert log[0]["template"] == "savings_rate" and log[0]["role"] == "rm"
    assert log[0]["statement_id"] == statement["id"] and log[0]["rows"] == 1
    audit = client.get(
        f"/statements/{statement['id']}/audit", headers=meridian_rm
    ).json()
    assert any(
        a["node"] == "warehouse.route" and a["event"] == "template" for a in audit
    )


def test_category_question_returns_a_table(client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    with patch("app.pipeline.chat.run_chat_turn", side_effect=_never_called):
        events = sse_events(
            client,
            "POST",
            f"/statements/{statement['id']}/chat",
            meridian_rm,
            {"question": "top 3 categories"},
        )
    assert events[0]["template"] == "top_categories" and events[0]["rows"] == 3
    assert "| category |" in events[-1]["data"]


def test_rm_is_refused_reviewer_queries_and_it_is_recorded(client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    with patch("app.pipeline.chat.run_chat_turn", side_effect=_never_called):
        events = sse_events(
            client,
            "POST",
            f"/statements/{statement['id']}/chat",
            meridian_rm,
            {"question": "Show me the pending reviews"},
        )
    assert events[0]["event"] == "denied" and events[0]["template"] == "pending_reviews"
    assert "reserved for reviewer" in events[-1]["data"]
    audit = client.get(
        f"/statements/{statement['id']}/audit", headers=meridian_rm
    ).json()
    assert any(
        a["event"] == "denied" and a["payload"]["template"] == "pending_reviews"
        for a in audit
    )
    # Nothing ran: the query log has no row for it.
    assert not any(
        q["template"] == "pending_reviews"
        for q in client.get("/warehouse/query-log", headers=meridian_rm).json()
    )


def test_reviewer_sees_the_queue_from_the_warehouse(
    client, meridian_rm, meridian_reviewer, mocked_llm
):
    # Put a decision in the queue: M-1003 over-declares.
    customer_id = _customer_by_ref(client, meridian_rm, "M-1003")
    statement = upload_sample(
        client,
        meridian_rm,
        customer_id,
        ROOT / "data" / "sample_3_cashflow_stressed.csv",
    )
    sse_events(client, "POST", f"/statements/{statement['id']}/run", meridian_rm)
    with patch("app.pipeline.chat.run_chat_turn", side_effect=_never_called):
        events = sse_events(
            client,
            "POST",
            f"/statements/{statement['id']}/chat",
            meridian_reviewer,
            {"question": "what needs review?"},
        )
    assert events[0]["template"] == "pending_reviews" and events[0]["rows"] >= 1
    assert "M-1003" in events[-1]["data"]


def test_warehouse_query_endpoint_is_role_checked_and_tenant_scoped(
    client, meridian_rm, harbor_rm
):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    ok = client.post(
        "/warehouse/query",
        headers=meridian_rm,
        json={"template": "savings_rate", "params": {"statement_id": statement["id"]}},
    )
    assert ok.status_code == 200 and ok.json()["rows"][0]["risk_band"] in {
        "Low",
        "Medium",
        "High",
    }
    # Same statement id, other bank's token: the view filters it out.
    other = client.post(
        "/warehouse/query",
        headers=harbor_rm,
        json={"template": "savings_rate", "params": {"statement_id": statement["id"]}},
    )
    assert other.status_code == 200 and other.json()["rows"] == []
    assert (
        client.post(
            "/warehouse/query",
            headers=meridian_rm,
            json={"template": "pending_reviews"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/warehouse/query", headers=meridian_rm, json={"template": "nope"}
        ).status_code
        == 404
    )
    listed = {
        t["name"]
        for t in client.get("/warehouse/templates", headers=meridian_rm).json()
    }
    assert "pending_reviews" not in listed and "savings_rate" in listed
    assert any(
        m["name"] == "risk_band"
        for m in client.get("/warehouse/metrics", headers=meridian_rm).json()
    )


def test_limit_is_clamped(client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    result = client.post(
        "/warehouse/query",
        headers=meridian_rm,
        json={
            "template": "top_categories",
            "params": {"statement_id": statement["id"], "limit": 100000},
        },
    ).json()
    assert result["params"]["limit"] == semantic.MAX_LIMIT


def test_product_question_still_goes_to_the_tool_chat(client, meridian_rm):
    customer_id = _customer_by_ref(client, meridian_rm, "M-1001")
    statement = upload_sample(client, meridian_rm, customer_id)
    seen = {}

    def fake_turn(question, history, metrics, df, tenant=None):
        from langchain_core.messages import AIMessage, HumanMessage

        seen["question"] = question
        yield "The sweep account suits a saver."
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content="The sweep account suits a saver."))

    with patch("app.pipeline.chat.run_chat_turn", side_effect=fake_turn):
        events = sse_events(
            client,
            "POST",
            f"/statements/{statement['id']}/chat",
            meridian_rm,
            {"question": "Which product would you recommend and why?"},
        )
    assert events[0]["event"] == "token" and seen["question"]
    assert login(client, "harbor", "rm")  # sanity: other tenants unaffected
