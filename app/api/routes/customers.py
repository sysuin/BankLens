"""Customers: the people whose statements get analysed."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.api import service
from app.api.deps import Principal, current_user, require_role
from app.api.schemas import CustomerCreate, CustomerOut

router = APIRouter(prefix="/customers", tags=["customers"])


@router.get("", response_model=list[CustomerOut])
async def list_customers(
    principal: Principal = Depends(current_user),
) -> list[CustomerOut]:
    rows = await service.list_customers(principal.tenant_id)
    return [CustomerOut(**row) for row in rows]


@router.post("", response_model=CustomerOut, status_code=status.HTTP_201_CREATED)
async def create_customer(
    body: CustomerCreate, principal: Principal = Depends(require_role("rm"))
) -> CustomerOut:
    try:
        row = await service.create_customer(
            principal.tenant_id,
            body.external_ref,
            body.full_name,
            body.declared_monthly_income,
        )
    except IntegrityError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"customer '{body.external_ref}' already exists"
        ) from exc
    return CustomerOut(**row)
