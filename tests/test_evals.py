"""
Layer 1 of the evaluation suite, run as part of the normal test suite.

These are evaluations rather than unit tests — they assert end-to-end behaviour
over a golden dataset rather than the behaviour of one function. They live here
because they need no API key and cost nothing, so there is no reason not to run
them on every push. The paid layer runs separately via evals/run_evals.py.
"""

import pytest

from app.pipeline.analyzer import compute_metrics
from evals.checks import run_deterministic_checks
from evals.dataset import (
    RISK_SWEEP,
    build_cases,
    categorize_rules_only,
    materialize,
)

CASES = build_cases()
CASE_IDS = [case.case_id for case in CASES]


@pytest.fixture(scope="module")
def metrics_by_case() -> dict:
    """Materialize and score every golden case once, then share the results."""
    computed = {}
    for case in CASES:
        frame = materialize(case)
        computed[case.case_id] = compute_metrics(categorize_rules_only(frame))
    return computed


class TestGoldenDataset:
    """The dataset itself must be sound before anything it asserts means much."""

    def test_dataset_is_not_trivially_small(self):
        """Three statements is a smoke test; an eval set needs spread."""
        assert len(CASES) >= 50

    def test_case_ids_are_unique(self):
        assert len(CASE_IDS) == len(set(CASE_IDS))

    def test_every_risk_band_is_represented(self):
        bands = {case.expected_risk for case in CASES}
        assert bands == {"Low", "Medium", "High"}

    def test_sweep_covers_both_sides_of_each_boundary(self):
        """A sweep that never straddles a boundary cannot detect it moving."""
        rates = [rate for rate, _ in RISK_SWEEP]
        for boundary in (10.0, 30.0):
            assert any(r < boundary for r in rates)
            assert any(r >= boundary for r in rates)

    def test_llm_sample_is_a_small_subset(self):
        """The paid layer must stay cheap — a handful of cases, not all of them."""
        sampled = [c for c in CASES if c.include_in_llm_eval]
        assert 0 < len(sampled) <= 10

    def test_deficit_cases_carry_the_credit_restriction(self):
        for case in CASES:
            if case.target_savings_rate is not None and case.target_savings_rate < 0:
                assert case.forbidden_products, f"{case.case_id} missing restriction"


class TestDeterministicEvals:
    """Layer 1 over the full golden set."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_case_passes_all_deterministic_checks(self, case, metrics_by_case):
        metrics = metrics_by_case[case.case_id]
        results = run_deterministic_checks(
            metrics, case.target_savings_rate, case.expected_risk
        )
        failures = [r for r in results if not r.passed and not r.advisory]
        assert not failures, "; ".join(f"{r.name}: {r.detail}" for r in failures)


class TestAggregateProperties:
    """Properties that only hold across the set, not within a single case."""

    def test_health_score_rises_with_savings_rate(self, metrics_by_case):
        """Across the whole set, a better savings rate must never score worse."""
        points = sorted(
            (m.savings_rate_pct, m.financial_health_score)
            for m in metrics_by_case.values()
        )
        scores = [score for _rate, score in points]
        assert scores == sorted(scores)

    def test_risk_never_improves_as_savings_rate_falls(self, metrics_by_case):
        order = {"High": 0, "Medium": 1, "Low": 2}
        points = sorted(
            (m.savings_rate_pct, order[m.risk_profile])
            for m in metrics_by_case.values()
        )
        ranks = [rank for _rate, rank in points]
        assert ranks == sorted(ranks)

    def test_every_deficit_case_is_flagged(self, metrics_by_case):
        for case_id, metrics in metrics_by_case.items():
            if metrics.savings_amount < 0:
                assert metrics.is_cashflow_negative, case_id

    def test_synthesis_preserves_transaction_count(self):
        """Rescaling amounts must not add or drop rows."""
        from evals.dataset import load_base

        for case in CASES:
            if case.literal_frame is not None:
                continue
            assert len(materialize(case)) == len(load_base(case.archetype))
