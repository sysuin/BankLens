"""Liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.db.session import app_engine
from app.pipeline.rag import list_tenants

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    database = "ok"
    try:
        async with app_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        database = f"error: {type(exc).__name__}"
    return HealthResponse(
        status="ok" if database == "ok" else "degraded",
        database=database,
        tenants_on_disk=list_tenants(),
        version=VERSION,
    )
