"""
BankLens evaluation runner.

Usage:
    python -m evals.run_evals                 # layer 1 only, free, no API key
    python -m evals.run_evals --with-llm      # adds the sampled paid layer
    python -m evals.run_evals --with-llm --judge   # adds groundedness judging

Exits non-zero if any blocking check fails, so it can gate a build. Advisory
checks are printed but never fail the run.
"""

import argparse
import sys
from collections import Counter

from app.core.logger import get_logger
from app.pipeline.analyzer import compute_metrics
from evals.checks import CheckResult, run_deterministic_checks, run_grounded_checks
from evals.dataset import EvalCase, build_cases, categorize_rules_only, materialize

logger = get_logger(__name__)


def evaluate_deterministic(case: EvalCase) -> list[CheckResult]:
    """Run layer 1 for a single case. No network, no cost."""
    frame = materialize(case)
    metrics = compute_metrics(categorize_rules_only(frame))
    return run_deterministic_checks(
        metrics, case.target_savings_rate, case.expected_risk
    )


def evaluate_grounded(case: EvalCase, use_judge: bool) -> list[CheckResult]:
    """Run layer 2 for a single case. Requires an API key; costs money."""
    from app.pipeline.agent import build_profile
    from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve

    frame = materialize(case)
    metrics = compute_metrics(categorize_rules_only(frame))

    # Same query builder the app uses — an eval that constructs its own query
    # measures a retrieval path that does not exist in production.
    chunks = retrieve(build_retrieval_query(metrics), build_vector_store())
    profile = build_profile(metrics, chunks)

    results = run_grounded_checks(profile, metrics, case.forbidden_products)

    if use_judge:
        from evals.judge import judge_groundedness

        verdict = judge_groundedness(profile, metrics, chunks)
        results.append(
            CheckResult(
                "judge_groundedness",
                verdict.grounded,
                verdict.reasoning
                + (
                    f" | unsupported: {verdict.unsupported_claims}"
                    if verdict.unsupported_claims
                    else ""
                ),
            )
        )
        results.append(
            CheckResult(
                "judge_respects_assigned_risk",
                not verdict.contradicts_assigned_risk,
                verdict.reasoning,
            )
        )

    return results


def golden_queries() -> list[tuple[str, str]]:
    """(query, expected risk band) for every case, using the production query builder."""
    from app.pipeline.rag import build_retrieval_query

    queries = []
    for case in build_cases():
        metrics = compute_metrics(categorize_rules_only(materialize(case)))
        queries.append((build_retrieval_query(metrics), metrics.risk_profile))
    return queries


def report_bm25_headroom() -> None:
    """
    Print how much a reranker could possibly recover, using the offline half.

    If the relevant document already sits inside the wider candidate pool but
    misses the final cut, reranking can recover it. If it is absent from the
    pool entirely, the problem is upstream and no reranker will fix it.
    """
    from app.core.config import settings
    from evals.retrieval import measure_bm25_headroom

    result = measure_bm25_headroom(
        golden_queries(),
        final_k=settings.retrieval_k,
        candidate_k=settings.retrieval_candidate_k,
    )

    print("\nBM25 headroom (lexical half only, no API calls)")
    print(f"{'':<14}{'hit':>8}{'mrr':>8}{'ndcg':>8}{'prec':>8}")
    for label, key in (
        (f"top-{settings.retrieval_k}", "final_k"),
        (f"top-{settings.retrieval_candidate_k}", "candidate_k"),
    ):
        means = result[key]
        print(
            f"{label:<14}{means['hit']:>8.3f}{means['mrr']:>8.3f}"
            f"{means['ndcg']:>8.3f}{means['precision']:>8.3f}"
        )
    gap = result["candidate_k"]["hit"] - result["final_k"]["hit"]
    print(f"{'headroom':<14}{gap:>8.3f}   (hit rate recoverable by reranking)")


def report_reranker_ab() -> None:
    """Print the reranker A/B over the golden queries. Requires an API key."""
    from evals.retrieval import compare_reranker

    result = compare_reranker(golden_queries())

    print(
        f"\nReranker A/B — backend='{result['backend']}', "
        f"{result['candidate_k']} candidates -> {result['final_k']} passages"
    )
    print(f"{'':<14}{'hit':>8}{'mrr':>8}{'ndcg':>8}{'prec':>8}{'harmful':>9}")
    for label, key in (("fusion only", "baseline"), ("reranked", "reranked")):
        means = result[key]
        print(
            f"{label:<14}{means['hit']:>8.3f}{means['mrr']:>8.3f}"
            f"{means['ndcg']:>8.3f}{means['precision']:>8.3f}{means['harmful']:>9.3f}"
        )
    delta = result["delta"]
    print(
        f"{'delta':<14}{delta['hit']:>+8.3f}{delta['mrr']:>+8.3f}"
        f"{delta['ndcg']:>+8.3f}{delta['precision']:>+8.3f}{delta['harmful']:>+9.3f}"
    )


def print_scorecard(rows: list[tuple[str, CheckResult]]) -> int:
    """Print a per-check summary and return the number of blocking failures."""
    by_check: dict[str, Counter] = {}
    for _case_id, result in rows:
        counter = by_check.setdefault(result.name, Counter())
        counter["pass" if result.passed else "fail"] += 1

    width = max(len(name) for name in by_check) if by_check else 20

    print("\n" + "=" * (width + 34))
    print(f"{'CHECK'.ljust(width)}   PASS   FAIL   RATE")
    print("=" * (width + 34))

    for name, counter in sorted(by_check.items()):
        total = counter["pass"] + counter["fail"]
        rate = counter["pass"] / total * 100 if total else 0.0
        print(
            f"{name.ljust(width)}   {counter['pass']:>4}   {counter['fail']:>4}   {rate:5.1f}%"
        )

    failures = [
        (case_id, result)
        for case_id, result in rows
        if not result.passed and not result.advisory
    ]
    advisories = [
        (case_id, result)
        for case_id, result in rows
        if not result.passed and result.advisory
    ]

    if advisories:
        print("\nADVISORY (not blocking):")
        for case_id, result in advisories[:10]:
            print(f"  - [{case_id}] {result.name}: {result.detail}")

    if failures:
        print("\nFAILURES:")
        for case_id, result in failures:
            print(f"  - [{case_id}] {result.name}: {result.detail}")
    else:
        print("\nAll blocking checks passed.")

    print()
    return len(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BankLens eval suite.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run the sampled grounded layer (requires OPENAI_API_KEY, costs money).",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Add LLM-as-judge groundedness scoring. Implies --with-llm.",
    )
    parser.add_argument(
        "--headroom",
        action="store_true",
        help="Report BM25 retrieval headroom. Free, no API key needed.",
    )
    parser.add_argument(
        "--retrieval-ab",
        action="store_true",
        help="Report reranker A/B over the golden queries (requires OPENAI_API_KEY).",
    )
    args = parser.parse_args()
    use_llm = args.with_llm or args.judge

    cases = build_cases()
    rows: list[tuple[str, CheckResult]] = []

    print(f"Running layer 1 (deterministic) over {len(cases)} cases...")
    for case in cases:
        for result in evaluate_deterministic(case):
            rows.append((case.case_id, result))

    if args.headroom:
        report_bm25_headroom()

    if args.retrieval_ab:
        report_reranker_ab()

    if use_llm:
        sampled = [c for c in cases if c.include_in_llm_eval]
        print(f"Running layer 2 (grounded) over {len(sampled)} sampled cases...")
        for case in sampled:
            try:
                for result in evaluate_grounded(case, use_judge=args.judge):
                    rows.append((case.case_id, result))
            except Exception as exc:  # noqa: BLE001 - a failed case is a result
                rows.append(
                    (case.case_id, CheckResult("grounded_layer_ran", False, str(exc)))
                )

    return 1 if print_scorecard(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
