"""
Seed the platform with two synthetic banks.

    python -m app.db.seed            # idempotent: re-running updates nothing that exists
    python -m app.db.seed --with-statements   # also ingest the sample statements

Tenants:
    meridian  Meridian Bank         catalogue: knowledge_base/meridian (10 products)
    harbor    Harbor Credit Union   catalogue: knowledge_base/harbor   (8 products)

Users per tenant (password for all: `banklens-demo`):
    rm@<tenant>.example        role rm
    reviewer@<tenant>.example  role reviewer

Customers per tenant carry a declared monthly income, which Phase 2 compares
with the income observed in their statements. Everything here is synthetic;
no real person, account or institution is represented. Refuses to run when
BANKLENS_ENV=production.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from app.core.config import settings
from app.core.logger import get_logger
from app.db.models import Customer, Role, Tenant, User
from app.db.session import admin_session, dispose_engines
from app.api.security import hash_password

logger = get_logger(__name__)

DEMO_PASSWORD = "banklens-demo"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TENANTS = [
    {"slug": "meridian", "name": "Meridian Bank"},
    {"slug": "harbor", "name": "Harbor Credit Union"},
]

USERS = [
    ("rm", "Relationship Manager", Role.rm),
    ("reviewer", "Credit Reviewer", Role.reviewer),
]

# (external_ref, name, declared monthly income, sample statement to ingest)
# Declared incomes are chosen so that some match the statement's observed
# income and some do not — the discrepancy cases for Phase 2.
CUSTOMERS = {
    "meridian": [
        # matches the statement (observed 200,000/month)
        ("M-1001", "Asha Verma", Decimal("200000"), "sample_1_high_saver.csv"),
        # matches (observed 95,000/month)
        ("M-1002", "Rohan Mehta", Decimal("95000"), "sample_2_active_spender.csv"),
        # over-declared: observed 50,000/month vs 120,000 declared -> review
        ("M-1003", "Priya Nair", Decimal("120000"), "sample_3_cashflow_stressed.csv"),
        ("M-1004", "Dev Kapoor", Decimal("55000"), None),
    ],
    "harbor": [
        # matches (observed 100,250 for the single month)
        ("H-2001", "Lena Fischer", Decimal("100000"), "sample_statement.csv"),
        # under-declared: observed 50,000/month vs 40,000 declared -> review
        ("H-2002", "Marco Ruiz", Decimal("40000"), "sample_3_cashflow_stressed.pdf"),
        ("H-2003", "Ines Duarte", Decimal("6200"), None),
    ],
}


async def _ensure_tenant(session, slug: str, name: str) -> Tenant:
    tenant = (
        await session.execute(select(Tenant).where(Tenant.slug == slug))
    ).scalar_one_or_none()
    if tenant is None:
        tenant = Tenant(id=uuid.uuid4(), slug=slug, name=name)
        session.add(tenant)
        await session.flush()
        logger.info("Created tenant %s (%s)", slug, tenant.id)
    return tenant


async def _ensure_user(
    session, tenant: Tenant, local: str, full_name: str, role: Role
) -> User:
    email = f"{local}@{tenant.slug}.example"
    user = (
        await session.execute(
            select(User).where(User.tenant_id == tenant.id, User.email == email)
        )
    ).scalar_one_or_none()
    if user is None:
        user = User(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            email=email,
            full_name=f"{full_name} ({tenant.name})",
            role=role,
            password_hash=hash_password(DEMO_PASSWORD),
        )
        session.add(user)
        await session.flush()
        logger.info("Created user %s role=%s", email, role.value)
    return user


async def _ensure_customer(
    session, tenant: Tenant, ref: str, name: str, declared: Decimal
) -> Customer:
    customer = (
        await session.execute(
            select(Customer).where(
                Customer.tenant_id == tenant.id, Customer.external_ref == ref
            )
        )
    ).scalar_one_or_none()
    if customer is None:
        customer = Customer(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            external_ref=ref,
            full_name=name,
            declared_monthly_income=declared,
        )
        session.add(customer)
        await session.flush()
        logger.info("Created customer %s/%s", tenant.slug, ref)
    return customer


def assert_seed_is_safe_for(env: str) -> None:
    """
    Refuse to seed a production database.

    The users this module creates share a password that is printed in the
    README and pre-filled in the console. That is right for a laptop demo
    and wrong anywhere real, so the seed follows the JWT-secret guard in
    `app/api/security.py`: in production it refuses before touching the
    database.
    """
    if env.strip().lower() == "production":
        raise RuntimeError(
            "BANKLENS_ENV=production: refusing to seed demo tenants and users "
            "with the published demo password."
        )


async def seed(with_statements: bool = False) -> dict:
    """Create tenants, users and customers. Returns ids for callers (tests)."""
    assert_seed_is_safe_for(settings.banklens_env)
    created: dict = {"tenants": {}, "users": {}, "customers": {}}
    async with admin_session() as session:
        for spec in TENANTS:
            tenant = await _ensure_tenant(session, spec["slug"], spec["name"])
            created["tenants"][tenant.slug] = tenant.id
            for local, full_name, role in USERS:
                user = await _ensure_user(session, tenant, local, full_name, role)
                created["users"][f"{local}@{tenant.slug}"] = user.id
            for ref, name, declared, _sample in CUSTOMERS[tenant.slug]:
                customer = await _ensure_customer(session, tenant, ref, name, declared)
                created["customers"][ref] = customer.id

    if with_statements:
        from app.api.service import ingest_statement_file

        for slug, rows in CUSTOMERS.items():
            tenant_id = created["tenants"][slug]
            rm_id = created["users"][f"rm@{slug}"]
            for ref, _name, _declared, sample in rows:
                if not sample:
                    continue
                path = DATA_DIR / sample
                if not path.exists():
                    logger.warning("Sample %s missing; skipping", path)
                    continue
                statement_id = await ingest_statement_file(
                    tenant_id=tenant_id,
                    tenant_slug=slug,
                    customer_id=created["customers"][ref],
                    uploaded_by=rm_id,
                    filename=path.name,
                    content=path.read_bytes(),
                )
                logger.info(
                    "Ingested %s for %s/%s -> %s", sample, slug, ref, statement_id
                )
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed BankLens with two synthetic banks."
    )
    parser.add_argument("--with-statements", action="store_true")
    args = parser.parse_args()

    async def run():
        try:
            result = await seed(with_statements=args.with_statements)
        finally:
            await dispose_engines()
        return result

    result = asyncio.run(run())
    print(
        f"Seeded {len(result['tenants'])} tenants, {len(result['users'])} users, "
        f"{len(result['customers'])} customers. Demo password: {DEMO_PASSWORD}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
