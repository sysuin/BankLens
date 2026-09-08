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

from app.db.session import dispose_engines
from app.warehouse.pilot import pilot_summary, render


async def _main(tenant: str) -> int:
    try:
        print(render(await pilot_summary(tenant)))
        return 0
    finally:
        await dispose_engines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot report from run timestamps.")
    parser.add_argument("--tenant", default="meridian")
    sys.exit(asyncio.run(_main(parser.parse_args().tenant)))
