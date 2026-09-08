"""
The review queue: pending decisions, and the reviewer's approve/reject.

Deciding resumes the checkpointed graph. The response streams the remaining
node events as Server-Sent Events, the same shape as `POST /statements/{id}/run`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api import decisions
from app.api.deps import Principal, current_user, require_role
from app.api.schemas import DecisionOut, ReviewRequest
from app.api.service import NotFound
from app.api.sse import sse_response
from app.db.models import DecisionStatus
from app.graph.builder import resume_run

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("", response_model=list[DecisionOut])
async def list_reviews(
    state: str = Query(
        "pending", pattern="^(pending|approved|rejected|auto_cleared|all)$"
    ),
    principal: Principal = Depends(current_user),
) -> list[DecisionOut]:
    wanted = None if state == "all" else DecisionStatus(state)
    rows = await decisions.list_decisions(principal.tenant_id, wanted)
    return [DecisionOut(**row) for row in rows]


@router.get("/{decision_id}", response_model=DecisionOut)
async def get_review(
    decision_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> DecisionOut:
    try:
        return DecisionOut(
            **await decisions.get_decision(principal.tenant_id, decision_id)
        )
    except NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "decision not found")


@router.post("/{decision_id}")
async def decide(
    decision_id: uuid.UUID,
    body: ReviewRequest,
    principal: Principal = Depends(require_role("reviewer")),
) -> StreamingResponse:
    try:
        decision = await decisions.get_decision(principal.tenant_id, decision_id)
    except NotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "decision not found")
    if decision["status"] != "pending":
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"decision already {decision['status']}"
        )
    action = "approved" if body.action == "approve" else "rejected"
    events = resume_run(
        tenant_id=principal.tenant_id,
        run_id=uuid.UUID(str(decision["run_id"])),
        action=action,
        note=body.note,
        reviewer_email=principal.email,
        reviewer_id=principal.user_id,
    )
    return sse_response(events)
