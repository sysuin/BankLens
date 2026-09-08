"""Read side of tracing: traces per statement, one trace as a tree."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import String, cast, func, select

from app.api.service import NotFound
from app.db.models import Span, Statement
from app.db.session import tenant_session


def _f(value) -> float | None:
    return float(value) if isinstance(value, (Decimal, int, float)) else None


async def list_traces(tenant_id: uuid.UUID, statement_id: uuid.UUID) -> list[dict]:
    """One row per trace touching the statement: root name, duration, cost, tokens."""
    async with tenant_session(tenant_id) as session:
        if await session.get(Statement, statement_id) is None:
            raise NotFound("statement not found")
        touching = select(Span.trace_id).where(Span.statement_id == statement_id)
        totals = (
            select(
                Span.trace_id,
                func.min(Span.start_time).label("started"),
                func.max(Span.end_time).label("ended"),
                func.coalesce(func.sum(Span.cost_usd), 0).label("cost"),
                func.coalesce(func.sum(Span.tokens_in), 0).label("tin"),
                func.coalesce(func.sum(Span.tokens_out), 0).label("tout"),
                func.count().label("spans"),
                func.max(cast(Span.run_id, String)).label("run_id"),
            )
            # Whole traces, not only the spans stamped with the statement id:
            # a model call inside the categorizer has no statement id yet.
            .where(Span.trace_id.in_(touching))
            .group_by(Span.trace_id)
            .order_by(func.min(Span.start_time).desc())
        )
        rows = (await session.execute(totals)).all()
        out = []
        for trace_id, started, ended, cost, tin, tout, spans, run_id in rows:
            root = (
                await session.execute(
                    select(Span.name)
                    .where(Span.trace_id == trace_id, Span.parent_span_id.is_(None))
                    .limit(1)
                )
            ).scalar_one_or_none()
            out.append(
                {
                    "trace_id": trace_id,
                    "root": root or "?",
                    "run_id": run_id,
                    "started": started,
                    "duration_ms": round((ended - started).total_seconds() * 1000, 1),
                    "cost_usd": _f(cost) or 0.0,
                    "tokens_in": int(tin),
                    "tokens_out": int(tout),
                    "spans": int(spans),
                }
            )
        return out


async def get_trace(tenant_id: uuid.UUID, trace_id: str) -> dict:
    """All spans of one trace, ordered by start, with depth for a waterfall."""
    async with tenant_session(tenant_id) as session:
        rows = (
            (
                await session.execute(
                    select(Span)
                    .where(Span.trace_id == trace_id)
                    .order_by(Span.start_time, Span.id)
                )
            )
            .scalars()
            .all()
        )
    if not rows:
        raise NotFound("trace not found")
    t0 = min(r.start_time for r in rows)
    by_id = {r.span_id: r for r in rows}

    def depth(r: Span) -> int:
        d, cur = 0, r
        while cur.parent_span_id and cur.parent_span_id in by_id:
            cur = by_id[cur.parent_span_id]
            d += 1
        return d

    spans = [
        {
            "span_id": r.span_id,
            "parent_span_id": r.parent_span_id,
            "name": r.name,
            "depth": depth(r),
            "offset_ms": round((r.start_time - t0).total_seconds() * 1000, 1),
            "duration_ms": _f(r.duration_ms),
            "status": r.status,
            "model": r.model,
            "tokens_in": r.tokens_in,
            "tokens_out": r.tokens_out,
            "cost_usd": _f(r.cost_usd),
            "attributes": {
                k: v
                for k, v in (r.attributes or {}).items()
                if not k.startswith("banklens.")
            },
        }
        for r in rows
    ]
    total_ms = max((s["offset_ms"] + (s["duration_ms"] or 0)) for s in spans)
    return {
        "trace_id": trace_id,
        "run_id": next((r.run_id for r in rows if r.run_id), None),
        "statement_id": next((r.statement_id for r in rows if r.statement_id), None),
        "total_ms": round(total_ms, 1),
        "cost_usd": round(sum(s["cost_usd"] or 0 for s in spans), 6),
        "tokens_in": sum(s["tokens_in"] or 0 for s in spans),
        "tokens_out": sum(s["tokens_out"] or 0 for s in spans),
        "spans": spans,
    }


def waterfall_lines(trace: dict, width: int = 40) -> list[str]:
    """A fixed-width text waterfall, for the CLI and the console."""
    total = max(trace["total_ms"], 1.0)
    lines = []
    for s in trace["spans"]:
        start = int(s["offset_ms"] / total * width)
        length = max(1, int((s["duration_ms"] or 0) / total * width))
        bar = " " * start + "█" * min(length, width - start)
        money = f"${s['cost_usd']:.4f}" if s["cost_usd"] else ""
        toks = (
            f"{s['tokens_in']}/{s['tokens_out']} tok"
            if s["tokens_in"] is not None
            else ""
        )
        name = ("  " * s["depth"]) + s["name"]
        lines.append(
            f"{name:<34} {bar:<{width}} {s['duration_ms'] or 0:>9.1f} ms  {toks:<14} {money}"
        )
    return lines
