"""
Real users on a deployment that must never run the seed.

The seed refuses production, so a deployed platform has tenants and no
people. `app/db/users.py` makes the first ones; these tests check that what
it creates can actually sign in, and that the ways an operator can get it
wrong are refused with a message rather than a traceback.
"""

from __future__ import annotations

import pytest

from app.db.users import UserError, check_password, create_user
from tests.conftest import run_db

GOOD_PASSWORD = "correct-horse-battery-staple"


def _create(pg_cluster, **kwargs):
    async def go(engine):
        return await create_user(engine=engine, **kwargs)

    return run_db(pg_cluster["admin_url"], go)


def test_a_created_user_can_sign_in_and_has_the_role_given(client, pg_cluster):
    result = _create(
        pg_cluster,
        tenant_slug="meridian",
        email="Asha.Verma@bank.example",
        full_name="Asha Verma",
        role="reviewer",
        password=GOOD_PASSWORD,
    )
    assert result == {
        "tenant": "meridian",
        "email": "asha.verma@bank.example",  # normalised
        "role": "reviewer",
        "action": "created",
    }

    response = client.post(
        "/auth/login",
        json={
            "tenant": "meridian",
            "email": "asha.verma@bank.example",
            "password": GOOD_PASSWORD,
        },
    )
    assert response.status_code == 200, response.text
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    # The role is real: a reviewer sees the queue, and may not start runs.
    assert client.get("/reviews", headers=headers).status_code == 200
    assert client.get("/customers", headers=headers).status_code == 200

    # The same person does not exist at the other bank.
    assert (
        client.post(
            "/auth/login",
            json={
                "tenant": "harbor",
                "email": "asha.verma@bank.example",
                "password": GOOD_PASSWORD,
            },
        ).status_code
        == 401
    )


def test_a_reset_password_replaces_the_old_one(client, pg_cluster):
    common = {
        "tenant_slug": "harbor",
        "email": "priya.rao@bank.example",
        "full_name": "Priya Rao",
        "role": "rm",
    }
    _create(pg_cluster, password=GOOD_PASSWORD, **common)
    with pytest.raises(UserError, match="already exists"):
        _create(pg_cluster, password=GOOD_PASSWORD, **common)

    new_password = "another-long-passphrase"
    result = _create(pg_cluster, password=new_password, reset=True, **common)
    assert result["action"] == "password reset"

    def login(password: str) -> int:
        return client.post(
            "/auth/login",
            json={
                "tenant": "harbor",
                "email": common["email"],
                "password": password,
            },
        ).status_code

    assert login(new_password) == 200
    assert login(GOOD_PASSWORD) == 401


@pytest.mark.parametrize(
    ("password", "message"),
    [
        ("short", "at least 12"),
        ("banklens-demo", "published"),
        ("  ChangeMe  ", "published"),
    ],
)
def test_weak_or_published_passwords_are_refused(password, message):
    with pytest.raises(UserError, match=message):
        check_password(password)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tenant_slug": "nosuchbank"}, "unknown tenant"),
        ({"email": "not-an-email"}, "not an email"),
        ({"role": "admin"}, "role must be one of"),
    ],
)
def test_operator_mistakes_are_named(pg_cluster, kwargs, message):
    args = {
        "tenant_slug": "meridian",
        "email": "someone@bank.example",
        "full_name": "Someone",
        "role": "rm",
        "password": GOOD_PASSWORD,
    }
    args.update(kwargs)
    with pytest.raises(UserError, match=message):
        _create(pg_cluster, **args)
