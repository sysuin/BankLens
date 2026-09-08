"""Phase 0 baseline: per-stage latency and token cost of one full BankLens analysis.

Runs the pipeline exactly as main.py does (sanitize -> categorize -> metrics ->
retrieve -> build_profile) on the sample statements, with the profile cache
disabled so every run pays the real price. Writes a JSON + Markdown report.

    PROFILE_CACHE_ENABLED=false python -m evals.baseline.measure_baseline --repeats 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pandas as pd
from langchain_community.callbacks.manager import get_openai_callback

from app.core.config import settings
from app.pipeline.agent import build_profile
from app.pipeline.analyzer import compute_metrics
from app.pipeline.categorizer import categorize_dataframe
from app.pipeline.pdf_parser import parse_pdf_statement
from app.pipeline.rag import build_retrieval_query, build_vector_store, retrieve
from app.pipeline.sanitizer import sanitize_dataframe

# USD per 1M tokens, OpenAI list prices as of Sep 2026 (update if they change).
PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}
SAMPLES = sorted(Path("data").glob("sample_*"))


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        out = fn(*a, **k)
    return out, time.perf_counter() - t0, cb.prompt_tokens, cb.completion_tokens


def cost(model: str, p: int, c: int) -> float:
    i, o = PRICES[model]
    return (p * i + c * o) / 1_000_000


def run_once(path: Path, vector_store) -> dict:
    row = {"file": path.name, "stages": {}}
    if path.suffix == ".pdf":
        df, dt, *_ = timed(parse_pdf_statement, str(path))
    else:
        df, dt, *_ = timed(pd.read_csv, path)
    row["stages"]["parse"] = {"s": dt}
    row["rows"] = int(len(df))

    df, dt, *_ = timed(sanitize_dataframe, df)
    row["stages"]["sanitize"] = {"s": dt}

    df, dt, p, c = timed(categorize_dataframe, df)
    row["stages"]["categorize"] = {"s": dt, "in": p, "out": c, "usd": cost("gpt-4o-mini", p, c)}
    row["llm_fallback_rows"] = int((df["category"] == "Others").sum())

    metrics, dt, *_ = timed(compute_metrics, df)
    row["stages"]["analyze"] = {"s": dt}

    chunks, dt, p, c = timed(retrieve, build_retrieval_query(metrics), vector_store)
    # multi-query rewrites use gpt-4o-mini; embeddings are negligible
    row["stages"]["retrieve"] = {"s": dt, "in": p, "out": c, "usd": cost("gpt-4o-mini", p, c)}

    profile, dt, p, c = timed(build_profile, metrics, chunks)
    row["stages"]["profile"] = {"s": dt, "in": p, "out": c, "usd": cost("gpt-4o", p, c)}

    row["total_s"] = sum(s["s"] for s in row["stages"].values())
    row["total_usd"] = sum(s.get("usd", 0.0) for s in row["stages"].values())
    row["risk"] = metrics.risk_profile.value if hasattr(metrics.risk_profile, "value") else str(metrics.risk_profile)
    row["products"] = [profile.recommended_product_1, profile.recommended_product_2] if hasattr(profile, "recommended_product_1") else None
    return row


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    assert not settings.profile_cache_enabled, "set PROFILE_CACHE_ENABLED=false"

    t0 = time.perf_counter()
    vs = build_vector_store()
    warm = time.perf_counter() - t0

    runs = []
    for r in range(args.repeats):
        for path in SAMPLES:
            try:
                row = run_once(path, vs)
                row["repeat"] = r
                runs.append(row)
                print(f"{path.name:<32} {row['total_s']:6.1f}s  ${row['total_usd']:.4f}  rows={row['rows']}")
            except Exception as e:  # keep going; record the failure
                runs.append({"file": path.name, "repeat": r, "error": f"{type(e).__name__}: {e}"})
                print(f"{path.name:<32} ERROR {type(e).__name__}: {e}")

    ok = [r for r in runs if "error" not in r]
    totals = [r["total_s"] for r in ok]
    usd = [r["total_usd"] for r in ok]
    stage_names = ["parse", "sanitize", "categorize", "analyze", "retrieve", "profile"]
    summary = {
        "n_runs": len(ok),
        "n_errors": len(runs) - len(ok),
        "vector_store_warm_s": warm,
        "total_s": {"p50": pct(totals, 0.5), "p95": pct(totals, 0.95), "min": min(totals), "max": max(totals)},
        "usd_per_statement": {"p50": pct(usd, 0.5), "mean": statistics.fmean(usd), "max": max(usd)},
        "stage_p50_s": {s: pct([r["stages"][s]["s"] for r in ok], 0.5) for s in stage_names},
        "stage_share_of_cost": {
            s: sum(r["stages"][s].get("usd", 0) for r in ok) / max(sum(usd), 1e-9) for s in stage_names
        },
        "tokens_p50": {
            s: {"in": pct([r["stages"][s].get("in", 0) for r in ok], 0.5), "out": pct([r["stages"][s].get("out", 0) for r in ok], 0.5)}
            for s in ("categorize", "retrieve", "profile")
        },
        "settings": {
            "model": settings.openai_model, "mini": settings.openai_mini_model,
            "rerank_backend": settings.rerank_backend, "multi_query": settings.multi_query_enabled,
            "retrieval_k": settings.retrieval_k, "candidate_k": settings.retrieval_candidate_k,
        },
    }
    out = Path("evals/baseline")
    out.mkdir(exist_ok=True)
    json.dump({"summary": summary, "runs": runs}, open(out / "baseline_runs.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
