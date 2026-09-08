"""
Print a trace as a waterfall in the terminal: where the time and money went.

    python -m scripts.show_trace --tenant harbor --email reviewer@harbor.example
    python -m scripts.show_trace --trace <trace_id> ...

With no --trace, shows the most recent graph run of the tenant. Talks to the
API (default http://127.0.0.1:8000), so it sees exactly what a user sees.
"""

from __future__ import annotations

import argparse
import os

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--api", default=os.environ.get("BANKLENS_API_URL", "http://127.0.0.1:8000")
    )
    ap.add_argument("--tenant", default="meridian")
    ap.add_argument("--email", default=None)
    ap.add_argument("--password", default="banklens-demo")
    ap.add_argument("--trace", default=None, help="trace id; default = latest run")
    args = ap.parse_args()
    email = args.email or f"rm@{args.tenant}.example"

    token = (
        httpx.post(
            f"{args.api}/auth/login",
            json={"tenant": args.tenant, "email": email, "password": args.password},
            timeout=30,
        )
        .raise_for_status()
        .json()["access_token"]
    )
    headers = {"Authorization": f"Bearer {token}"}

    trace_id = args.trace
    if trace_id is None:
        statements = httpx.get(
            f"{args.api}/statements", headers=headers, timeout=30
        ).json()
        candidates = []
        for s in statements:
            for t in httpx.get(
                f"{args.api}/statements/{s['id']}/traces", headers=headers, timeout=30
            ).json():
                if t["run_id"]:
                    candidates.append((t["started"], t["trace_id"], s["customer_name"]))
        if not candidates:
            print("no graph runs traced yet for this tenant")
            return 1
        candidates.sort(reverse=True)
        _, trace_id, who = candidates[0]
        print(f"latest run: {who}")

    trace = httpx.get(f"{args.api}/traces/{trace_id}", headers=headers, timeout=30)
    if trace.status_code == 404:
        print("trace not found (or not yours)")
        return 1
    body = trace.json()
    print(f"trace {body['trace_id']}  run {body['run_id']}")
    print(
        f"total {body['total_ms']:.0f} ms · {body['tokens_in']} in / {body['tokens_out']} out tokens · "
        f"${body['cost_usd']:.4f}\n"
    )
    for line in body["waterfall"]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
