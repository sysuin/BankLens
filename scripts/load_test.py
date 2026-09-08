"""
Bulk load: enqueue N statements, run the worker, report throughput and cost.

    make load                      # 50 ingest jobs, worker inline, Meridian
    python -m scripts.load_test --n 50 --kind ingest_and_run --tenant harbor

Statements are generated from the sample CSVs with small random perturbations
(amounts ±10 %, shuffled rows), so the fifty are distinct without inventing a
new statement format. `ingest` costs nothing (the deterministic half only);
`ingest_and_run` runs the graph and pays for the model, unless the gateway is
on the local provider.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import random
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [ROOT / "data" / n for n in ("sample_1_high_saver.csv", "sample_2_active_spender.csv", "sample_3_cashflow_stressed.csv", "sample_statement.csv")]


def synth_statement(seed: int) -> tuple[str, bytes]:
    rng = random.Random(seed)
    src = SAMPLES[seed % len(SAMPLES)]
    df = pd.read_csv(src)
    df["amount"] = (df["amount"] * rng.uniform(0.9, 1.1)).round(2)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return f"load_{seed:03d}_{src.stem}.csv", buf.getvalue().encode("utf-8")


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.environ.get("BANKLENS_API_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--tenant", default="meridian")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kind", default="ingest", choices=["ingest", "ingest_and_run"])
    ap.add_argument("--no-worker", action="store_true", help="enqueue only; a worker is already running")
    args = ap.parse_args()

    token = httpx.post(
        f"{args.api}/auth/login",
        json={"tenant": args.tenant, "email": f"rm@{args.tenant}.example", "password": "banklens-demo"},
        timeout=30,
    ).raise_for_status().json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    customers = httpx.get(f"{args.api}/customers", headers=headers, timeout=30).json()

    t0 = time.perf_counter()
    job_ids = []
    for i in range(args.n):
        name, content = synth_statement(i)
        customer = customers[i % len(customers)]
        job = httpx.post(
            f"{args.api}/jobs",
            headers=headers,
            data={"customer_id": customer["id"], "kind": args.kind},
            files={"file": (name, content, "text/csv")},
            timeout=60,
        ).raise_for_status().json()
        job_ids.append(job["id"])
    enqueue_s = time.perf_counter() - t0
    print(f"enqueued {args.n} {args.kind} jobs in {enqueue_s:.1f}s")

    if not args.no_worker:
        from app.worker import run_worker

        t1 = time.perf_counter()
        processed = asyncio.run(run_worker(once=True))
        work_s = time.perf_counter() - t1
        print(f"worker processed {processed} jobs in {work_s:.1f}s")

    jobs = {j["id"]: j for j in httpx.get(f"{args.api}/jobs", headers=headers, timeout=60).json()}
    mine = [jobs[j] for j in job_ids if j in jobs]
    done = [j for j in mine if j["status"] in ("done", "awaiting_review")]
    failed = [j for j in mine if j["status"] == "failed"]
    durations = [j["duration_ms"] for j in done if j["duration_ms"] is not None]
    cost = sum(j["cost_usd"] or 0 for j in done)
    summary = httpx.get(f"{args.api}/jobs/summary", headers=headers, timeout=30).json()

    print()
    print(f"{'finished':<24}{len(done)}/{args.n}   failed {len(failed)}")
    if durations:
        print(f"{'per-statement p50':<24}{percentile(durations, 0.5):>8.0f} ms")
        print(f"{'per-statement p95':<24}{percentile(durations, 0.95):>8.0f} ms")
    if summary.get("statements_per_minute"):
        print(f"{'throughput':<24}{summary['statements_per_minute']:>8.1f} statements/min (wall clock, all finished jobs)")
    print(f"{'cost':<24}${cost:.4f} total   ${cost / max(len(done), 1):.4f} per statement")
    for j in failed[:5]:
        print("  failed:", j["filename"], j["error"])
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
