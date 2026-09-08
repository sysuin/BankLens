"""Login, tokens and role gates."""

from __future__ import annotations

import time

import jwt

from app.core.config import settings
from tests.conftest import login


def test_health_reports_database_and_tenants(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert {"meridian", "harbor"} <= set(body["tenants_on_disk"])


def test_login_returns_a_token_with_tenant_and_role(client):
    response = client.post(
        "/auth/login",
        json={
            "tenant": "meridian",
            "email": "rm@meridian.example",
            "password": "banklens-demo",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tenant"] == "meridian" and body["role"] == "rm"
    claims = jwt.decode(body["access_token"], settings.jwt_secret, algorithms=["HS256"])
    assert claims["tenant"] == "meridian" and claims["role"] == "rm"
    assert claims["exp"] > time.time()


def test_wrong_password_and_unknown_tenant_are_both_401(client):
    bad_password = client.post(
        "/auth/login",
        json={"tenant": "meridian", "email": "rm@meridian.example", "password": "nope"},
    )
    unknown_tenant = client.post(
        "/auth/login",
        json={
            "tenant": "nowhere",
            "email": "rm@meridian.example",
            "password": "banklens-demo",
        },
    )
    assert bad_password.status_code == 401
    assert unknown_tenant.status_code == 401
    # Same message for both: do not reveal which part was wrong.
    assert bad_password.json()["detail"] == unknown_tenant.json()["detail"]


def test_me_returns_the_principal(client, meridian_reviewer):
    body = client.get("/auth/me", headers=meridian_reviewer).json()
    assert body["tenant"] == "meridian"
    assert body["tenant_name"] == "Meridian Bank"
    assert body["role"] == "reviewer"


def test_requests_without_a_token_are_rejected(client):
    assert client.get("/customers").status_code == 401
    assert client.get("/statements").status_code == 401
    assert (
        client.get(
            "/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
        ).status_code
        == 401
    )


def test_expired_token_is_rejected(client):
    from app.api.security import create_access_token
    import uuid

    token = create_access_token(
        user_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        tenant_slug="meridian",
        role="rm",
        email="x@meridian.example",
        expires_minutes=-1,
    )
    response = client.get("/customers", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "expired" in response.json()["detail"]


def test_reviewer_cannot_upload_or_create_customers(client, meridian_reviewer):
    create = client.post(
        "/customers",
        headers=meridian_reviewer,
        json={
            "external_ref": "M-9999",
            "full_name": "Nobody",
            "declared_monthly_income": 1,
        },
    )
    assert create.status_code == 403
    with open("data/sample_statement.csv", "rb") as fh:
        upload = client.post(
            "/statements",
            headers=meridian_reviewer,
            data={"customer_id": "00000000-0000-0000-0000-000000000000"},
            files={"file": ("sample_statement.csv", fh, "text/csv")},
        )
    assert upload.status_code == 403


def test_reviewer_can_read(client, meridian_reviewer):
    assert client.get("/customers", headers=meridian_reviewer).status_code == 200
    assert client.get("/statements", headers=meridian_reviewer).status_code == 200


def test_every_response_carries_a_request_id(client):
    response = client.get("/health")
    assert response.headers["x-request-id"]
    echoed = client.get("/health", headers={"x-request-id": "trace-me-123"})
    assert echoed.headers["x-request-id"] == "trace-me-123"


def test_login_helper_works_for_both_banks(client):
    assert login(client, "meridian", "rm")
    assert login(client, "harbor", "reviewer")
