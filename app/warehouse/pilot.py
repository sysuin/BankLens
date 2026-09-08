"""
What a pilot would measure, from the timestamps the platform already keeps.

The discovery document's "about twenty minutes per statement" is an
assumption. This module does not replace it with another guess: it reports
the numbers the database can prove (how long the platform took, how long a
reviewer took, how many statements a day) and prints the assumption next
to them, labelled, so a real pilot only has to swap one line.

Runs as the API role inside a tenant-pinned transaction, so the report is
one bank's, by policy rather than by filter.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.session import tenant_session, unscoped_app_session

MANUAL_MINUTES_ASSUMED = 20.0  # docs/discovery.md, "Assumption"

_SQL = {
    "statements": "SELECT count(*) FROM statements",
    "statements_per_day": """
        SELECT coalesce(avg(n), 0) FROM (
            SELECT count(*) AS n FROM statements GROUP BY created_at::date
        ) d
    """,
    "runs_completed": "SELECT count(*) FROM runs WHERE status = 'completed'",
    "runs_failed": "SELECT count(*) FROM runs WHERE status = 'failed'",
    # Platform time only: a run that paused for a reviewer has the wait
    # subtracted, so this measures the machine and the next row the human.
    "seconds_to_profile": """
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY s),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY s)
        FROM (
            SELECT extract(epoch FROM (r.updated_at - r.created_at))
                   - coalesce(extract(epoch FROM (d.decided_at - d.created_at)), 0) AS s
            FROM runs r
            LEFT JOIN decisions d ON d.run_id = r.id AND d.decided_at IS NOT NULL
            WHERE r.status = 'completed'
        ) t
    """,
    "reviews_decided": """
        SELECT count(*) FROM decisions WHERE status IN ('approved', 'rejected')
    """,
    "reviews_pending": "SELECT count(*) FROM decisions WHERE status = 'pending'",
    "reviewer_wait_seconds": """
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY s),
               percentile_cont(0.95) WITHIN GROUP (ORDER BY s)
        FROM (
            SELECT extract(epoch FROM (decided_at - created_at)) AS s
            FROM decisions WHERE decided_at IS NOT NULL
        ) t
    """,
    "cache_hits": "SELECT coalesce(sum(hits), 0) FROM profile_cache",
    "model_cost_usd": """
        SELECT coalesce(sum((attributes->>'llm.cost_usd')::numeric), 0)
        FROM spans WHERE attributes ? 'llm.cost_usd'
    """,
}


async def _tenant_id(slug: str, engine: AsyncEngine | None) -> uuid.UUID:
    async with unscoped_app_session(engine) as session:
        row = (
            await session.execute(
                text("SELECT id FROM tenants WHERE slug = :slug"), {"slug": slug}
            )
        ).first()
    if row is None:
        raise LookupError(f"unknown tenant '{slug}'")
    return row[0]


async def pilot_summary(
    tenant_slug: str, engine: AsyncEngine | None = None
) -> dict[str, Any]:
    """Every number a pilot report needs, for one bank."""
    tenant_id = await _tenant_id(tenant_slug, engine)
    out: dict[str, Any] = {"tenant": tenant_slug}
    async with tenant_session(tenant_id, engine) as session:
        for key, sql in _SQL.items():
            row = (await session.execute(text(sql))).first()
            if key in ("seconds_to_profile", "reviewer_wait_seconds"):
                out[f"{key}_p50"] = (
                    float(row[0]) if row and row[0] is not None else None
                )
                out[f"{key}_p95"] = (
                    float(row[1]) if row and row[1] is not None else None
                )
            else:
                out[key] = float(row[0]) if row and row[0] is not None else 0.0
    out["manual_minutes_assumed"] = MANUAL_MINUTES_ASSUMED
    p50 = out.get("seconds_to_profile_p50")
    out["minutes_saved_if_assumption_holds"] = (
        round(MANUAL_MINUTES_ASSUMED - p50 / 60.0, 2) if p50 is not None else None
    )
    return out


def render(summary: dict[str, Any]) -> str:
    """A plain table for the terminal."""

    def sec(v):
        return "n/a" if v is None else f"{v:,.1f} s"

    rows = [
        ("statements", f"{summary['statements']:.0f}"),
        ("statements per active day", f"{summary['statements_per_day']:.1f}"),
        (
            "runs completed / failed",
            f"{summary['runs_completed']:.0f} / {summary['runs_failed']:.0f}",
        ),
        (
            "seconds to profile p50 / p95",
            f"{sec(summary['seconds_to_profile_p50'])} / {sec(summary['seconds_to_profile_p95'])}",
        ),
        (
            "reviews decided / pending",
            f"{summary['reviews_decided']:.0f} / {summary['reviews_pending']:.0f}",
        ),
        (
            "reviewer wait p50 / p95",
            f"{sec(summary['reviewer_wait_seconds_p50'])} / {sec(summary['reviewer_wait_seconds_p95'])}",
        ),
        ("profile cache hits", f"{summary['cache_hits']:.0f}"),
        ("model spend, all time", f"${summary['model_cost_usd']:.4f}"),
        (
            "manual minutes per statement",
            f"{summary['manual_minutes_assumed']:.0f} (ASSUMPTION, docs/discovery.md)",
        ),
        (
            "minutes saved per statement",
            (
                "n/a (no completed runs)"
                if summary["minutes_saved_if_assumption_holds"] is None
                else f"{summary['minutes_saved_if_assumption_holds']:.1f} if the assumption holds"
            ),
        ),
    ]
    width = max(len(k) for k, _ in rows) + 2
    lines = [f"Pilot report — tenant '{summary['tenant']}'", "-" * (width + 40)]
    lines += [f"{k:<{width}}{v}" for k, v in rows]
    lines.append("-" * (width + 40))
    lines.append(
        "Measured rows come from runs, decisions, profile_cache and spans. "
        "The manual figure is not measured; a pilot replaces it with a stopwatch."
    )
    return "\n".join(lines)
