"""
Evaluation checks for BankLens, split by what they cost to run.

Layer 1 (deterministic) needs no API key and no network. It asserts the things
the application decides for itself: metric arithmetic, risk banding, health
score behaviour, and the cashflow flag. It is free, so it runs on every push.

Layer 2 (grounded) requires a generated profile, and therefore an API key. It
asserts the things the model influences: whether retrieval surfaced the right
product, whether the recommendation is a product that exists, whether the
credit guardrail held, and whether the prose is grounded in the metrics.

Every check returns a CheckResult rather than raising, so a run produces a
scorecard instead of stopping at the first failure.
"""

import re
from dataclasses import dataclass

from app.pipeline.analyzer import FinancialMetrics
from app.pipeline.agent import CustomerProfile, resolve_product


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check against a single case."""

    name: str
    passed: bool
    detail: str = ""

    # Advisory checks are reported but do not fail the run. Used where a
    # heuristic is informative but not reliable enough to gate a build on.
    advisory: bool = False


# ── Layer 1: deterministic ────────────────────────────────────────────────────


def check_savings_rate_hits_target(
    metrics: FinancialMetrics, target: float | None, tolerance: float = 0.5
) -> CheckResult:
    """The synthesized statement should land on the savings rate it was built for."""
    if target is None:
        return CheckResult("savings_rate_target", True, "not a synthesized case")

    delta = abs(metrics.savings_rate_pct - target)
    return CheckResult(
        "savings_rate_target",
        delta <= tolerance,
        f"target={target:.2f}% actual={metrics.savings_rate_pct:.2f}% delta={delta:.3f}",
    )


def check_risk_band(metrics: FinancialMetrics, expected_risk: str) -> CheckResult:
    """The risk rating must match the hand-written expectation for this case."""
    return CheckResult(
        "risk_band",
        metrics.risk_profile == expected_risk,
        f"expected={expected_risk} actual={metrics.risk_profile} "
        f"(savings_rate={metrics.savings_rate_pct:.2f}%)",
    )


def check_health_score_range(metrics: FinancialMetrics) -> CheckResult:
    """The score must stay inside its declared bounds."""
    score = metrics.financial_health_score
    return CheckResult("health_score_range", 0 <= score <= 100, f"score={score}")


def check_cashflow_flag(metrics: FinancialMetrics) -> CheckResult:
    """The deficit flag must agree with the arithmetic it summarizes."""
    expected = metrics.savings_amount < 0
    return CheckResult(
        "cashflow_flag",
        metrics.is_cashflow_negative == expected,
        f"savings_amount={metrics.savings_amount:.2f} flag={metrics.is_cashflow_negative}",
    )


def run_deterministic_checks(
    metrics: FinancialMetrics, target: float | None, expected_risk: str
) -> list[CheckResult]:
    """Every layer 1 check for one case."""
    return [
        check_savings_rate_hits_target(metrics, target),
        check_risk_band(metrics, expected_risk),
        check_health_score_range(metrics),
        check_cashflow_flag(metrics),
    ]


# ── Layer 2: grounded ─────────────────────────────────────────────────────────


def check_products_are_real(profile: CustomerProfile) -> CheckResult:
    """
    Both recommended products must exist in the knowledge base.

    The schema validator already enforces this at parse time, so a failure here
    means the guardrail itself regressed.
    """
    unknown = [
        name
        for name in (profile.primary_product, profile.secondary_product)
        if resolve_product(name) is None
    ]
    return CheckResult(
        "products_are_real",
        not unknown,
        f"unrecognized={unknown}" if unknown else "both products resolve",
    )


def check_credit_guardrail(
    profile: CustomerProfile, forbidden_products: frozenset[str]
) -> CheckResult:
    """
    A customer in deficit must not be pushed further into unsecured credit.

    This is the check with the most real-world weight in the suite: it encodes
    a lending policy, not an implementation detail.
    """
    if not forbidden_products:
        return CheckResult("credit_guardrail", True, "no restriction for this case")

    violations = []
    for field_name in ("primary_product", "secondary_product"):
        name = getattr(profile, field_name)
        resolved = resolve_product(name)
        if resolved in forbidden_products:
            violations.append(f"{field_name}={name!r} -> {resolved}")

    return CheckResult(
        "credit_guardrail",
        not violations,
        "; ".join(violations) if violations else "no forbidden product recommended",
    )


def check_retrieval_surfaced_recommendation(profile: CustomerProfile) -> CheckResult:
    """
    The recommended product should appear among the retrieved sources.

    Grounding means the recommendation came *from* the retrieved context. A
    product recommended without its document being retrieved is the model
    drawing on parametric memory instead.
    """
    resolved = resolve_product(profile.primary_product)
    return CheckResult(
        "retrieval_supports_recommendation",
        resolved in set(profile.retrieved_sources),
        f"primary={profile.primary_product!r} -> {resolved} "
        f"retrieved={profile.retrieved_sources}",
    )


def check_sources_present(profile: CustomerProfile) -> CheckResult:
    """A profile with no sources cannot be explained to a customer."""
    return CheckResult(
        "sources_present",
        bool(profile.retrieved_sources),
        f"sources={profile.retrieved_sources}",
    )


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def check_percentages_supported(
    profile: CustomerProfile, metrics: FinancialMetrics, tolerance: float = 1.0
) -> CheckResult:
    """
    Percentages quoted in the prose should trace back to a computed metric.

    Advisory rather than blocking: a model may legitimately quote a figure from
    a product document ("6.5% per annum") that is not a customer metric, so a
    miss here is a prompt to look rather than proof of a fabrication.
    """
    supported = {
        round(metrics.savings_rate_pct, 2),
        round(metrics.expense_to_income_ratio * 100, 2),
        round(100 - metrics.savings_rate_pct, 2),
        float(metrics.financial_health_score),
    }

    prose = " ".join(
        [
            profile.income_stability_analysis,
            profile.spending_pattern_breakdown,
            profile.credit_risk_assessment,
            profile.primary_reason,
            profile.secondary_reason,
        ]
    )

    unmatched = [
        quoted
        for quoted in (float(m) for m in _PERCENT_RE.findall(prose))
        if not any(abs(quoted - known) <= tolerance for known in supported)
    ]

    return CheckResult(
        "percentages_supported",
        not unmatched,
        f"unmatched={unmatched} known={sorted(supported)}",
        advisory=True,
    )


def run_grounded_checks(
    profile: CustomerProfile,
    metrics: FinancialMetrics,
    forbidden_products: frozenset[str],
) -> list[CheckResult]:
    """Every layer 2 check that does not require a judge model."""
    return [
        check_products_are_real(profile),
        check_credit_guardrail(profile, forbidden_products),
        check_retrieval_surfaced_recommendation(profile),
        check_sources_present(profile),
        check_percentages_supported(profile, metrics),
    ]
