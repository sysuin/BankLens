"""
The decision graph: wiring, checkpointing, and the two entry points.

    start_run(...)   create a Run row, execute the graph from the top, and
                     stream node events; stops at the review interrupt
    resume_run(...)  feed a reviewer's answer into the checkpointed run and
                     stream the remaining nodes

Checkpoints live in Postgres through LangGraph's AsyncPostgresSaver, keyed
by thread id = run id. That is what makes "stop the server, come back
tomorrow, approve, and it finishes" work: nothing about the run is held in
memory. The checkpointer connects as the owner role because its tables are
its own; the run id that addresses them is only reachable through the
RLS-protected `runs` table.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool

from app.core.logger import get_logger
from app.db.models import Run, RunStatus
from app.db.session import _resolve_urls, tenant_session
from app.graph import audit, nodes
from app.graph.state import GraphState

logger = get_logger(__name__)

_pool: AsyncConnectionPool | None = None
_saver: AsyncPostgresSaver | None = None
_lock = asyncio.Lock()


def _psycopg_conninfo(sqlalchemy_url: str) -> str:
    """postgresql+asyncpg://u:p@h/db?host=/sock → postgresql://u:p@h/db?host=/sock"""
    return sqlalchemy_url.replace("postgresql+asyncpg://", "postgresql://", 1).replace(
        "postgresql+psycopg://", "postgresql://", 1
    )


async def get_checkpointer() -> AsyncPostgresSaver:
    global _pool, _saver
    async with _lock:
        if _saver is None:
            _, admin_url = _resolve_urls()
            _pool = AsyncConnectionPool(
                conninfo=_psycopg_conninfo(admin_url),
                min_size=1,
                max_size=4,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            await _pool.open()
            _saver = AsyncPostgresSaver(_pool)
            await _saver.setup()
            logger.info("LangGraph Postgres checkpointer ready")
    return _saver


async def close_checkpointer() -> None:
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool = _saver = None


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("load_context", nodes.load_context)
    graph.add_node("verify_income", nodes.verify_income)
    graph.add_node("await_review", nodes.await_review)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("narrate", nodes.narrate)
    graph.add_node("guardrails", nodes.guardrails)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("finalize_rejected", nodes.finalize_rejected)
    graph.add_node("finalize_blocked", nodes.finalize_blocked)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "verify_income")
    graph.add_conditional_edges(
        "verify_income",
        nodes.route_after_verify,
        {"await_review": "await_review", "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "await_review",
        nodes.route_after_review,
        {"retrieve": "retrieve", "finalize_rejected": "finalize_rejected"},
    )
    graph.add_edge("retrieve", "narrate")
    graph.add_edge("narrate", "guardrails")
    graph.add_conditional_edges(
        "guardrails",
        nodes.route_after_guardrails,
        {"finalize": "finalize", "finalize_blocked": "finalize_blocked"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("finalize_rejected", END)
    graph.add_edge("finalize_blocked", END)
    return graph


async def compiled():
    return build_graph().compile(checkpointer=await get_checkpointer())


def _config(run_id: str | uuid.UUID) -> dict:
    return {"configurable": {"thread_id": str(run_id)}}


async def _stream(app, payload, config) -> AsyncIterator[dict[str, Any]]:
    """Translate LangGraph 'updates' into small, serialisable events."""
    async for update in app.astream(payload, config, stream_mode="updates"):
        for node, value in update.items():
            if node == "__interrupt__":
                interrupts = value if isinstance(value, (list, tuple)) else [value]
                for item in interrupts:
                    yield {"event": "interrupt", "data": getattr(item, "value", item)}
                continue
            data = {}
            if isinstance(value, dict):
                keep = (
                    "observed_monthly_income",
                    "declared_monthly_income",
                    "discrepancy_pct",
                    "review_required",
                    "review_decision",
                    "from_cache",
                    "tokens_in",
                    "tokens_out",
                    "guardrail_violations",
                    "guardrail_warnings",
                    "profile_id",
                    "outcome",
                    "error",
                )
                data = {k: value[k] for k in keep if k in value}
                if "chunks" in value:
                    data["sources"] = sorted({c["source"] for c in value["chunks"]})
                if "profile" in value:
                    data["primary_product"] = value["profile"].get("primary_product")
                    data["secondary_product"] = value["profile"].get(
                        "secondary_product"
                    )
            yield {"event": "node", "node": node, "data": data}


async def start_run(
    *,
    tenant: str,
    tenant_id: uuid.UUID,
    statement_id: uuid.UUID,
    actor_email: str,
    actor_id: uuid.UUID | None,
) -> AsyncIterator[dict[str, Any]]:
    run_id = uuid.uuid4()
    async with tenant_session(tenant_id) as session:
        session.add(
            Run(
                id=run_id,
                tenant_id=tenant_id,
                statement_id=statement_id,
                started_by=actor_id,
                status=RunStatus.running,
                current_node="start",
            )
        )
    await audit.record(
        tenant_id,
        run_id=run_id,
        statement_id=statement_id,
        node="start",
        event="run_started",
        actor=actor_email,
    )
    yield {"event": "run", "run_id": str(run_id)}

    app = await compiled()
    initial: GraphState = {
        "tenant": tenant,
        "tenant_id": str(tenant_id),
        "statement_id": str(statement_id),
        "run_id": str(run_id),
        "actor": actor_email,
    }
    try:
        async for event in _stream(app, initial, _config(run_id)):
            yield event
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised as an event
        logger.exception("run %s failed", run_id)
        await audit.set_run_status(
            tenant_id, run_id, RunStatus.failed, error=f"{type(exc).__name__}: {exc}"
        )
        await audit.record(
            tenant_id,
            run_id=run_id,
            statement_id=statement_id,
            node="graph",
            event="failed",
            actor="system",
            payload={"error": f"{type(exc).__name__}: {exc}"},
        )
        yield {"event": "error", "data": f"{type(exc).__name__}: {exc}"}
        return
    yield {"event": "done", "run_id": str(run_id)}


async def resume_run(
    *,
    tenant_id: uuid.UUID,
    run_id: uuid.UUID,
    action: str,
    note: str | None,
    reviewer_email: str,
    reviewer_id: uuid.UUID | None,
) -> AsyncIterator[dict[str, Any]]:
    app = await compiled()
    answer = {
        "action": action,
        "note": note,
        "reviewer": reviewer_email,
        "reviewer_id": str(reviewer_id) if reviewer_id else None,
    }
    try:
        async for event in _stream(app, Command(resume=answer), _config(run_id)):
            yield event
    except Exception as exc:  # noqa: BLE001
        logger.exception("resume %s failed", run_id)
        await audit.set_run_status(
            tenant_id, run_id, RunStatus.failed, error=f"{type(exc).__name__}: {exc}"
        )
        yield {"event": "error", "data": f"{type(exc).__name__}: {exc}"}
        return
    yield {"event": "done", "run_id": str(run_id)}


async def pending_interrupt(run_id: uuid.UUID) -> dict | None:
    """The interrupt payload a checkpointed run is waiting on, if any."""
    app = await compiled()
    snapshot = await app.aget_state(_config(run_id))
    for task in getattr(snapshot, "tasks", ()):
        for item in getattr(task, "interrupts", ()):
            return getattr(item, "value", None)
    return None
