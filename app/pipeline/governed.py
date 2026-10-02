"""
The steps every front door must run, in one place.

The console's direct mode, the API, the bulk worker and the MCP server all
analyse statements. Two safeguards used to live only on the API path — the
statement injection scan at ingest and the decision graph's guardrails node —
so production (the console in direct mode) and the MCP tools skipped them.
Doctrine 5 says not to fork logic into a front door; this module is where the
shared path lives, and each door calls it:

    prepare_statement(df)                 mask -> scan -> categorise -> compute
    checked_profile(metrics, chunks, ..)  narrate (cached) -> check_profile

Neither touches the database, so the doors without one can run them. Doors
that have a database add their own audit rows around these calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.core.context import current_tenant
from app.pipeline.analyzer import FinancialMetrics, compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.sanitizer import sanitize_dataframe
from app.platform.guardrails import (
    ProfileCheck,
    StatementScan,
    check_profile,
    scan_statement,
)
from app.platform.tracing import span


@dataclass
class PreparedStatement:
    categorized: pd.DataFrame
    metrics: FinancialMetrics
    scan: StatementScan


def prepare_statement(raw: pd.DataFrame) -> PreparedStatement:
    """Mask PII, neutralise instruction-like rows, categorise, compute."""
    with span("pipeline.sanitize"):
        sanitized = sanitize_dataframe(raw)
    with span("guardrail.statement_scan") as current:
        # The statement is attacker-controlled text: neutralise
        # instruction-like descriptions before the categoriser or any model
        # sees them. Numbers are untouched.
        sanitized, scan = scan_statement(sanitized)
        current.set_attribute("guardrail.flagged_rows", scan.flagged_rows)
        current.set_attribute("guardrail.families", sorted(set(scan.families)))
    with span("pipeline.categorize") as current:
        categorized = categorize_dataframe(sanitized)
        current.set_attribute(
            "pipeline.llm_fallback_rows",
            int((categorized["category"] == "Others").sum()),
        )
    with span("pipeline.metrics") as current:
        metrics = compute_metrics(categorized)
        current.set_attribute("metrics.rows", int(metrics.transaction_count))
    return PreparedStatement(categorized, metrics, scan)


@dataclass
class CheckedProfile:
    profile: object | None  # CustomerProfile, or None when blocked
    check: ProfileCheck
    from_cache: bool


def check(
    profile, metrics: FinancialMetrics, chunks: list[dict], tenant: str | None = None
) -> ProfileCheck:
    """Run the profile guardrails on an already-built profile."""
    with span("guardrail.profile") as current:
        result = check_profile(
            profile.model_dump(mode="json"),
            metrics.model_dump(mode="json"),
            chunks,
            tenant or current_tenant(),
        )
        current.set_attribute("guardrail.blocked", result.blocked)
        current.set_attribute("guardrail.violations", len(result.violations))
        current.set_attribute("guardrail.warnings", len(result.warnings))
    return result


def checked_profile(
    metrics: FinancialMetrics, chunks: list[dict], tenant: str | None = None
) -> CheckedProfile:
    """Narrate (through the cache), then check; a blocked profile is withheld."""
    from app.pipeline.cache import cached_build_profile

    profile, from_cache = cached_build_profile(metrics, chunks, tenant=tenant)
    result = check(profile, metrics, chunks, tenant)
    return CheckedProfile(None if result.blocked else profile, result, from_cache)
