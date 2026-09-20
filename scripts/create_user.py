"""
Create the first real users on a deployed platform.

    python -m scripts.create_user --tenant meridian \\
        --email asha.verma@bank.example --name "Asha Verma" --role rm

The password is asked for on the terminal, twice, and never appears in an
argument, an environment variable, a log line or this repository. Needs the
owner database URL (DATABASE_ADMIN_URL), so run it where migrations run: on
the host, in a one-off container with the migration env file.

    docker run --rm -it --network banklens \\
      --env-file /home/ec2-user/banklens-platform/migrate.env \\
      <image> python -m scripts.create_user --tenant meridian \\
      --email asha.verma@bank.example --name "Asha Verma" --role rm

Add --reset-password to set a new password for someone who exists.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

from app.db.session import dispose_engines
from app.db.users import UserError, check_password, create_user


def _read_password() -> str:
    first = getpass.getpass("New password: ")
    check_password(first)
    if first != getpass.getpass("Repeat password: "):
        raise UserError("the two passwords do not match")
    return first


async def _main(args: argparse.Namespace, password: str) -> int:
    try:
        result = await create_user(
            tenant_slug=args.tenant,
            email=args.email,
            full_name=args.name,
            role=args.role,
            password=password,
            reset=args.reset_password,
        )
    finally:
        await dispose_engines()
    print(
        f"{result['action']}: {result['email']} "
        f"(role {result['role']}, tenant {result['tenant']})"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a BankLens user.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True, choices=["rm", "reviewer"])
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="Set a new password for a user who already exists.",
    )
    parsed = parser.parse_args()
    try:
        secret = _read_password()
        sys.exit(asyncio.run(_main(parsed, secret)))
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
