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

import csv
import statistics
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.session import tenant_session, unscoped_app_session

MANUAL_MINUTES_ASSUMED = 20.0  # docs/discovery.md, "Assumption"

# Stopwatch timings from the pilot sessions (docs/pilot_protocol.md). One row
# per statement: mode "manual" is the RM working from the ledger, "assisted"
# is the RM working from the BankLens profile. Real sessions go in
# data/pilot/timings.csv; data/pilot/timings.example.csv shows the columns.
TIMINGS_COLUMNS = (
    "session_date",
    "rm_id",
    "tenant",
    "statement_ref",
    "statement_lines",
    "mode",
    "minutes",
    "notes",
)


def load_timings(path: Path, tenant: str) -> dict[str, list[float]]:
    """{"manual": [...], "assisted": [...]} for one tenant, validated."""
    out: dict[str, list[float]] = {"manual": [], "assisted": []}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in TIMINGS_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for line, row in enumerate(reader, start=2):
            if row["tenant"].strip() != tenant:
                continue
            mode = row["mode"].strip().lower()
            if mode not in out:
                raise ValueError(f"{path}:{line}: mode must be manual or assisted")
            minutes = float(row["minutes"])
            if not 0 < minutes < 480:
                raise ValueError(f"{path}:{line}: minutes {minutes} is not plausible")
            out[mode].append(minutes)
    return out


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


def apply_timings(
    out: dict[str, Any], timings: dict[str, list[float]] | None
) -> dict[str, Any]:
    """Add stopwatch medians and, only when both modes were timed, minutes saved."""
    timings = timings or {"manual": [], "assisted": []}
    manual, assisted = timings.get("manual", []), timings.get("assisted", [])
    out["manual_sessions"] = len(manual)
    out["assisted_sessions"] = len(assisted)
    out["manual_minutes_measured"] = (
        round(statistics.median(manual), 2) if manual else None
    )
    out["assisted_minutes_measured"] = (
        round(statistics.median(assisted), 2) if assisted else None
    )
    # Only a like-for-like comparison counts as measured savings: both modes
    # timed by stopwatch. Anything less is reported, and labelled, as partial.
    out["minutes_saved_measured"] = (
        round(out["manual_minutes_measured"] - out["assisted_minutes_measured"], 2)
        if manual and assisted
        else None
    )
    return out


async def pilot_summary(
    tenant_slug: str,
    engine: AsyncEngine | None = None,
    timings: dict[str, list[float]] | None = None,
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

    return apply_timings(out, timings)


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
    ]
    measured_manual = summary.get("manual_minutes_measured")
    measured_assisted = summary.get("assisted_minutes_measured")
    saved = summary.get("minutes_saved_measured")
    if measured_manual is not None:
        rows.append(
            (
                "manual minutes per statement",
                f"{measured_manual:.1f} (MEASURED, median of "
                f"{summary['manual_sessions']} timed statements)",
            )
        )
    else:
        rows.append(
            (
                "manual minutes per statement",
                f"{summary['manual_minutes_assumed']:.0f} (ASSUMPTION, docs/discovery.md)",
            )
        )
    if measured_assisted is not None:
        rows.append(
            (
                "assisted minutes per statement",
                f"{measured_assisted:.1f} (MEASURED, median of "
                f"{summary['assisted_sessions']} timed statements)",
            )
        )
    if saved is not None:
        rows.append(("minutes saved per statement", f"{saved:.1f} (MEASURED)"))
    elif summary["minutes_saved_if_assumption_holds"] is None:
        rows.append(("minutes saved per statement", "n/a (no completed runs)"))
    else:
        rows.append(
            (
                "minutes saved per statement",
                f"{summary['minutes_saved_if_assumption_holds']:.1f} if the assumption holds",
            )
        )
    width = max(len(k) for k, _ in rows) + 2
    lines = [f"Pilot report — tenant '{summary['tenant']}'", "-" * (width + 40)]
    lines += [f"{k:<{width}}{v}" for k, v in rows]
    lines.append("-" * (width + 40))
    if saved is not None:
        lines.append(
            "Measured rows come from runs, decisions, profile_cache and spans; "
            "manual and assisted minutes come from stopwatch sessions "
            "(docs/pilot_protocol.md)."
        )
    else:
        lines.append(
            "Measured rows come from runs, decisions, profile_cache and spans. "
            "Minutes saved needs both manual and assisted stopwatch timings; "
            "see docs/pilot_protocol.md."
        )
    return "\n".join(lines)
