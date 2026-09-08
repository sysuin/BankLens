"""
Tenant isolation — the Phase 1 proof.

Two banks share one database and one API process. These tests show that a
principal from one bank cannot read, list, profile or chat about the other
bank's statements, by id or otherwise, and that the guarantee lives in the
database (row-level security), not only in the API's WHERE clauses.

To watch the guard fail on purpose (interview demo step 8):

    psql banklens -c "ALTER TABLE statements DISABLE ROW LEVEL SECURITY"
    pytest tests/test_tenancy.py::test_rls_hides_other_tenant_rows_from_a_pinned_session

That test goes red. Re-enable RLS and it goes green again.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select, text

from tests.conftest import run_db, upload_sample

# ── API level ────────────────────────────────────────────────────────────────


def test_each_bank_sees_only_its_own_customers(client, meridian_rm, harbor_rm):
    meridian = {
        c["external_ref"] for c in client.get("/customers", headers=meridian_rm).json()
    }
    harbor = {
        c["external_ref"] for c in client.get("/customers", headers=harbor_rm).json()
    }
    assert meridian and harbor
    assert meridian.isdisjoint(harbor)
    assert all(ref.startswith("M-") for ref in meridian)
    assert all(ref.startswith("H-") for ref in harbor)


def test_statement_by_id_is_invisible_across_tenants(client, meridian_rm, harbor_rm):
    customer_id = client.get("/customers", headers=meridian_rm).json()[0]["id"]
    statement = upload_sample(client, meridian_rm, customer_id)

    assert (
        client.get(f"/statements/{statement['id']}", headers=meridian_rm).status_code
        == 200
    )
    # Same id, other bank's token: not 403 (which would confirm existence) but 404.
    assert (
        client.get(f"/statements/{statement['id']}", headers=harbor_rm).status_code
        == 404
    )
    assert (
        client.post(
            f"/statements/{statement['id']}/profile", headers=harbor_rm
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/statements/{statement['id']}/chat",
            headers=harbor_rm,
            json={"question": "what is the savings rate?"},
        ).status_code
        == 404
    )
    ids_seen_by_harbor = {
        s["id"] for s in client.get("/statements", headers=harbor_rm).json()
    }
    assert statement["id"] not in ids_seen_by_harbor


def test_upload_against_another_banks_customer_is_rejected(
    client, meridian_rm, harbor_rm
):
    harbor_customer = client.get("/customers", headers=harbor_rm).json()[0]["id"]
    with open("data/sample_statement.csv", "rb") as fh:
        response = client.post(
            "/statements",
            headers=meridian_rm,
            data={"customer_id": harbor_customer},
            files={"file": ("sample_statement.csv", fh, "text/csv")},
        )
    assert response.status_code == 404


def test_token_from_one_bank_cannot_log_in_users_of_another(client):
    # A Meridian email with the Harbor tenant: invisible, therefore invalid.
    response = client.post(
        "/auth/login",
        json={
            "tenant": "harbor",
            "email": "rm@meridian.example",
            "password": "banklens-demo",
        },
    )
    assert response.status_code == 401


# ── Database level ───────────────────────────────────────────────────────────


def test_rls_hides_other_tenant_rows_from_a_pinned_session(
    pg_cluster, client, meridian_rm
):
    """
    The API role, pinned to Harbor, reads Meridian's statements: nothing.
    The same role pinned to Meridian: the row is there.
    """
    from app.db.models import Statement
    from app.db.session import tenant_session

    customer_id = client.get("/customers", headers=meridian_rm).json()[0]["id"]
    statement = upload_sample(client, meridian_rm, customer_id)
    statement_id = uuid.UUID(statement["id"])
    ids = pg_cluster["ids"]["tenants"]

    def read_as(tenant_id):
        async def _go(engine):
            async with tenant_session(tenant_id, engine=engine) as session:
                result = await session.execute(
                    select(Statement).where(Statement.id == statement_id)
                )
                return result.scalar_one_or_none()

        return run_db(pg_cluster["app_url"], _go)

    assert read_as(ids["meridian"]) is not None
    assert read_as(ids["harbor"]) is None


def test_unpinned_session_sees_no_tenant_rows_at_all(pg_cluster, client, meridian_rm):
    """Forgetting to pin the tenant is a bug that reads as 'empty', never as 'everything'."""
    from app.db.models import Customer, Statement
    from app.db.session import unscoped_app_session

    customer_id = client.get("/customers", headers=meridian_rm).json()[0]["id"]
    upload_sample(client, meridian_rm, customer_id)

    async def count_unpinned(engine):
        async with unscoped_app_session(engine=engine) as session:
            statements = (await session.execute(select(Statement))).scalars().all()
            customers = (await session.execute(select(Customer))).scalars().all()
            return len(statements), len(customers)

    assert run_db(pg_cluster["app_url"], count_unpinned) == (0, 0)


def test_rls_is_enforced_in_the_database_not_the_orm(pg_cluster):
    """Raw SQL, no ORM, pinned to Harbor: Meridian's customers do not exist."""
    ids = pg_cluster["ids"]["tenants"]

    async def raw(engine):
        async with engine.connect() as conn:
            async with conn.begin():
                await conn.execute(
                    text("SELECT set_config('app.tenant_id', :t, true)"),
                    {"t": str(ids["harbor"])},
                )
                rows = await conn.execute(
                    text("SELECT external_ref FROM customers WHERE tenant_id = :m"),
                    {"m": str(ids["meridian"])},
                )
                return rows.fetchall()

    assert run_db(pg_cluster["app_url"], raw) == []


def test_policies_exist_on_every_tenant_table(pg_cluster):
    """A new tenant table without a policy would silently be world-readable."""
    from app.db.models import TENANT_TABLES

    async def policies(engine):
        async with engine.connect() as conn:
            rows = await conn.execute(
                text(
                    "SELECT c.relname, c.relrowsecurity FROM pg_class c "
                    "JOIN pg_policy p ON p.polrelid = c.oid"
                )
            )
            return {name: enabled for name, enabled in rows.fetchall()}

    found = run_db(pg_cluster["admin_url"], policies)
    for table in TENANT_TABLES:
        assert found.get(table) is True, f"{table} has no active row-level policy"


@pytest.mark.parametrize("tenant", ["meridian", "harbor"])
def test_product_catalogue_is_per_tenant(tenant):
    from app.pipeline.agent import get_product_catalogue, resolve_product

    catalogue = get_product_catalogue(tenant)
    assert catalogue
    if tenant == "meridian":
        assert "credit_card.md" in catalogue
        assert (
            resolve_product("Harbor Cash-Back Credit Card", "harbor")
            == "cashback_credit_card.md"
        )
        assert resolve_product("Recurring Deposit", "harbor") is None
    else:
        assert "cashback_credit_card.md" in catalogue
        assert "credit_card.md" not in catalogue
        assert (
            resolve_product("Recurring Deposit", "meridian") == "recurring_deposit.md"
        )
