"""
Print the pilot report for one tenant.

    make pilot TENANT=meridian
    python -m scripts.pilot_report --tenant harbor

Every number except the manual-minutes assumption is read from the
database; the assumption is printed as one, so nobody quotes it as measured.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.session import dispose_engines
from app.warehouse.pilot import load_timings, pilot_summary, render

DEFAULT_TIMINGS = (
    Path(__file__).resolve().parent.parent / "data" / "pilot" / "timings.csv"
)


async def _main(tenant: str, timings_path: Path) -> int:
    timings = load_timings(timings_path, tenant) if timings_path.exists() else None
    try:
        print(render(await pilot_summary(tenant, timings=timings)))
        return 0
    finally:
        await dispose_engines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot report from run timestamps.")
    parser.add_argument("--tenant", default="meridian")
    parser.add_argument(
        "--timings",
        type=Path,
        default=DEFAULT_TIMINGS,
        help="Stopwatch timings CSV (see data/pilot/timings.example.csv).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args.tenant, args.timings)))
