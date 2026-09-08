"""Traces and the prompt registry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.api import traces
from app.api.deps import Principal, current_user
from app.api.service import NotFound
from app.platform import registry

router = APIRouter(tags=["tracing"])


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str, principal: Principal = Depends(current_user)
) -> dict:
    if len(trace_id) != 32:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trace not found")
    try:
        trace = await traces.get_trace(principal.tenant_id, trace_id)
    except NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "trace not found")
    trace["waterfall"] = traces.waterfall_lines(trace)
    return trace


@router.get("/platform/prompts")
async def prompt_versions(principal: Principal = Depends(current_user)) -> list[dict]:
    return await registry.list_versions()
