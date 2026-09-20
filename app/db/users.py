"""
Create and re-password real users, for a deployment with no seed.

`app/db/seed.py` refuses to run in production, and rightly: its users share a
password that is printed in the README. A deployed platform therefore starts
with tenants but no people, and this is how the first ones are made.

The password never reaches this module from a file, an argument or a log: the
script that wraps it (`scripts/create_user.py`) reads it from the operator's
terminal. What is stored is a bcrypt hash. Runs as the owner role, because it
writes a row no tenant session may write.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.security import hash_password
from app.core.logger import get_logger
from app.db.models import Role, Tenant, User
from app.db.session import admin_session

logger = get_logger(__name__)

MIN_PASSWORD_LENGTH = 12
# The seed's password is published; refuse it even where the seed cannot run.
FORBIDDEN_PASSWORDS = frozenset({"banklens-demo", "password", "changeme"})


class UserError(Exception):
    """Something the operator can fix: unknown tenant, weak password, clash."""


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if password.strip().lower() in FORBIDDEN_PASSWORDS:
        raise UserError("that password is published in this repository; pick another")


async def create_user(
    *,
    tenant_slug: str,
    email: str,
    full_name: str,
    role: str,
    password: str,
    reset: bool = False,
    engine: AsyncEngine | None = None,
) -> dict:
    """
    Create one user, or set an existing user's password when `reset` is true.

    Returns what is safe to print: tenant, email, role, and what happened.
    """
    check_password(password)
    email = email.strip().lower()
    if "@" not in email:
        raise UserError(f"{email!r} is not an email address")
    try:
        role_value = Role(role)
    except ValueError as exc:
        raise UserError(
            f"role must be one of {[r.value for r in Role]}, not {role!r}"
        ) from exc

    async with admin_session(engine) as session:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.slug == tenant_slug))
        ).scalar_one_or_none()
        if tenant is None:
            known = (await session.execute(select(Tenant.slug))).scalars().all()
            raise UserError(f"unknown tenant {tenant_slug!r}; known: {sorted(known)}")

        user = (
            await session.execute(
                select(User).where(User.tenant_id == tenant.id, User.email == email)
            )
        ).scalar_one_or_none()
        if user is not None and not reset:
            raise UserError(
                f"{email} already exists at {tenant_slug}; "
                "pass --reset-password to set a new password"
            )

        if user is None:
            user = User(
                id=uuid.uuid4(),
                tenant_id=tenant.id,
                email=email,
                full_name=full_name,
                role=role_value,
                password_hash=hash_password(password),
            )
            session.add(user)
            action = "created"
        else:
            user.password_hash = hash_password(password)
            user.full_name = full_name or user.full_name
            user.role = role_value
            action = "password reset"
        await session.flush()
        logger.info("%s user %s role=%s tenant=%s", action, email, role, tenant_slug)
        return {
            "tenant": tenant_slug,
            "email": email,
            "role": role_value.value,
            "action": action,
        }
