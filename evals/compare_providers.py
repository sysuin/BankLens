"""
The same golden suite against two providers, side by side.

    python -m evals.compare_providers                 # openai vs ollama
    python -m evals.compare_providers --providers ollama   # zero-key run

For each provider: the deterministic layer (identical by construction, it
never calls a model), then the sampled grounded layer with pass rate per
check, p50/p95 latency and mean cost per query. The point is not that the
local model wins; it is that the same suite runs against both and the
difference is a number you can read, not a guess.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from app.core.config import settings
from app.core.logger import get_logger
from evals import run_evals
from evals.dataset import build_cases

logger = get_logger(__name__)


def _reset_pipeline_caches() -> None:
    """Switching provider means different embeddings and a different cache key."""
    from app.pipeline import rag
    from app.platform import gateway

    rag.reset_bm25_cache()
    gateway.reset_circuits()
    gateway.invalidate_budget_cache()


def run_for(provider: str, use_judge: bool) -> dict:
    settings.llm_provider = provider
    settings.llm_fallback_enabled = False  # measure this provider, not its fallback
    _reset_pipeline_caches()
    run_evals.LATENCY_MS.clear()
    run_evals.COST_USD.clear()

    cases = build_cases()
    sampled = [c for c in cases if c.include_in_llm_eval]
    passed: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    errors = 0
    for case in sampled:
        try:
            for result in run_evals.evaluate_grounded(case, use_judge=use_judge):
                total[result.name] += 1
                passed[result.name] += int(result.passed)
        except Exception as exc:  # noqa: BLE001 - a failed case is a result
            errors += 1
            logger.warning("%s: %s failed: %s", provider, case.case_id, exc)
    n = len(run_evals.LATENCY_MS)
    return {
        "provider": provider,
        "model": settings.ollama_model if provider == "ollama" else settings.openai_model,
        "cases": len(sampled),
        "errors": errors,
        "checks": {k: (passed[k], total[k]) for k in sorted(total)},
        "p50_ms": run_evals.percentile(run_evals.LATENCY_MS, 0.5) if n else None,
        "p95_ms": run_evals.percentile(run_evals.LATENCY_MS, 0.95) if n else None,
        "cost_per_query": (sum(run_evals.COST_USD) / n) if n else 0.0,
    }


def print_table(rows: list[dict]) -> None:
    names = sorted({k for r in rows for k in r["checks"]})
    width = 34
    header = f"{'':<{width}}" + "".join(f"{r['provider'] + ' (' + r['model'] + ')':>26}" for r in rows)
    print("\nGolden suite, grounded layer, side by side\n")
    print(header)
    print("-" * len(header))
    for name in names:
        line = f"{name:<{width}}"
        for r in rows:
            p, t = r["checks"].get(name, (0, 0))
            line += f"{(str(p) + '/' + str(t) + ' (' + f'{100 * p / t:.0f}%' + ')') if t else 'n/a':>26}"
        print(line)
    print("-" * len(header))
    for label, key, fmt in (
        ("cases with errors", "errors", "{:d}"),
        ("p50 latency (ms)", "p50_ms", "{:.0f}"),
        ("p95 latency (ms)", "p95_ms", "{:.0f}"),
        ("cost per query (USD)", "cost_per_query", "{:.4f}"),
    ):
        line = f"{label:<{width}}"
        for r in rows:
            value = r.get(key)
            line += f"{(fmt.format(value) if value is not None else 'n/a'):>26}"
        print(line)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Golden suite against several providers.")
    ap.add_argument("--providers", default="openai,ollama")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    rows = []
    for provider in providers:
        if provider == "openai" and not settings.openai_api_key:
            print("openai skipped: no OPENAI_API_KEY", file=sys.stderr)
            continue
        print(f"running grounded layer on {provider}…", file=sys.stderr)
        rows.append(run_for(provider, args.judge))
    if not rows:
        print("nothing to compare", file=sys.stderr)
        return 1
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
