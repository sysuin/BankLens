"""Login and identity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import Principal, current_user
from app.api.schemas import LoginRequest, MeResponse, TokenResponse
from app.api.security import create_access_token, verify_password
from app.core.config import settings
from app.core.logger import get_logger
from app.db.models import Tenant, User
from app.db.session import tenant_session, unscoped_app_session

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)

_BAD_CREDENTIALS = HTTPException(
    status.HTTP_401_UNAUTHORIZED, "invalid tenant, email or password"
)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    # Step 1: the tenant, from the one table with no row-level policy.
    async with unscoped_app_session() as session:
        tenant = (
            await session.execute(
                select(Tenant).where(Tenant.slug == body.tenant.lower())
            )
        ).scalar_one_or_none()
    if tenant is None:
        raise _BAD_CREDENTIALS

    # Step 2: the user, inside that tenant's scope. A valid email from a
    # different bank is invisible here, which is the point.
    async with tenant_session(tenant.id) as session:
        user = (
            await session.execute(
                select(User).where(
                    User.tenant_id == tenant.id, User.email == body.email.lower()
                )
            )
        ).scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        logger.info("Login failed tenant=%s email=%s", body.tenant, body.email)
        raise _BAD_CREDENTIALS

    token = create_access_token(
        user_id=user.id,
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        role=user.role.value,
        email=user.email,
    )
    logger.info(
        "Login ok tenant=%s email=%s role=%s", tenant.slug, user.email, user.role.value
    )
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expire_minutes * 60,
        tenant=tenant.slug,
        role=user.role.value,
        email=user.email,
        full_name=user.full_name,
    )


@router.get("/me", response_model=MeResponse)
async def me(principal: Principal = Depends(current_user)) -> MeResponse:
    async with tenant_session(principal.tenant_id) as session:
        user = await session.get(User, principal.user_id)
        tenant = await session.get(Tenant, principal.tenant_id)
    if user is None or tenant is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")
    return MeResponse(
        user_id=user.id,
        tenant=tenant.slug,
        tenant_name=tenant.name,
        role=user.role.value,
        email=user.email,
        full_name=user.full_name,
    )
