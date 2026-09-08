"""
Statements: upload, inspect, profile, and chat.

Upload runs the deterministic half of the pipeline (parse → sanitize →
categorize → metrics) and stores the result. Profile runs the probabilistic
half (retrieve → narrate) on demand and stores that too, with the model and
prompt version that produced it. Chat streams over Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.api import decisions, service
from app.api.deps import Principal, current_user, require_role
from app.api.schemas import (
    AuditEventOut,
    ChatRequest,
    ProfileOut,
    RunOut,
    StatementDetail,
    StatementSummary,
)
from app.api.sse import sse_response
from app.core.context import tenant_scope
from app.core.logger import get_logger
from app.graph.builder import start_run

router = APIRouter(prefix="/statements", tags=["statements"])
logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "statement not found")


@router.post("", response_model=StatementSummary, status_code=status.HTTP_201_CREATED)
async def upload_statement(
    customer_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    principal: Principal = Depends(require_role("rm")),
) -> StatementSummary:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "statement over 5 MB"
        )
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    try:
        statement_id = await service.ingest_statement_file(
            tenant_id=principal.tenant_id,
            tenant_slug=principal.tenant,
            customer_id=customer_id,
            uploaded_by=principal.user_id,
            filename=file.filename or "statement",
            content=content,
            uploaded_by_email=principal.email,
        )
    except service.NotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "customer not found") from exc
    except (service.BadInput, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    detail = await service.get_statement(principal.tenant_id, statement_id)
    return StatementSummary(**detail)


@router.get("", response_model=list[StatementSummary])
async def list_statements(
    principal: Principal = Depends(current_user),
) -> list[StatementSummary]:
    rows = await service.list_statements(principal.tenant_id)
    return [StatementSummary(**row) for row in rows]


@router.get("/{statement_id}", response_model=StatementDetail)
async def get_statement(
    statement_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> StatementDetail:
    try:
        detail = await service.get_statement(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()
    return StatementDetail(**detail)


@router.post("/{statement_id}/profile", response_model=ProfileOut)
async def create_profile(
    statement_id: uuid.UUID, principal: Principal = Depends(require_role("rm"))
) -> ProfileOut:
    try:
        row = await service.generate_profile(
            principal.tenant_id, principal.tenant, statement_id
        )
    except service.NotFound:
        raise _not_found()
    return ProfileOut(**row)


@router.get("/{statement_id}/profile", response_model=ProfileOut)
async def get_profile(
    statement_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> ProfileOut:
    try:
        row = await service.latest_profile(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no profile generated yet")
    return ProfileOut(**row)


# ── Decision graph ───────────────────────────────────────────────────────────


@router.post("/{statement_id}/run")
async def run_graph(
    statement_id: uuid.UUID, principal: Principal = Depends(require_role("rm"))
) -> StreamingResponse:
    """
    Run the decision graph for a statement, streaming node events as SSE.

    Events: `run` (the run id), `node` (a node finished, with a small data
    payload), `interrupt` (paused for a reviewer; the run is checkpointed),
    `done`, `error`. A paused run is resumed from `POST /reviews/{id}`.
    """
    try:
        await service.get_statement(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()
    events = start_run(
        tenant=principal.tenant,
        tenant_id=principal.tenant_id,
        statement_id=statement_id,
        actor_email=principal.email,
        actor_id=principal.user_id,
    )
    return sse_response(events)


@router.get("/{statement_id}/runs", response_model=list[RunOut])
async def list_runs(
    statement_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> list[RunOut]:
    try:
        rows = await decisions.list_runs(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()
    return [RunOut(**row) for row in rows]


@router.get("/{statement_id}/traces")
async def statement_traces(
    statement_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> list[dict]:
    from app.api import traces

    try:
        return await traces.list_traces(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()


@router.get("/{statement_id}/audit", response_model=list[AuditEventOut])
async def statement_audit(
    statement_id: uuid.UUID, principal: Principal = Depends(current_user)
) -> list[AuditEventOut]:
    try:
        rows = await decisions.list_audit(
            principal.tenant_id, statement_id=statement_id
        )
    except service.NotFound:
        raise _not_found()
    return [AuditEventOut(**row) for row in rows]


@router.post("/{statement_id}/chat")
async def chat(
    statement_id: uuid.UUID,
    body: ChatRequest,
    principal: Principal = Depends(current_user),
) -> StreamingResponse:
    """
    Stream an answer as Server-Sent Events.

    Events: `token` (a text chunk), `done` (the full answer), `error`.
    The chat module is synchronous and streams a generator; it runs in a
    worker thread and hands chunks to the event loop through a queue.
    """
    try:
        metrics, df = await service.chat_context(principal.tenant_id, statement_id)
    except service.NotFound:
        raise _not_found()

    history = [
        (
            HumanMessage(content=m.content)
            if m.role == "user"
            else AIMessage(content=m.content)
        )
        for m in body.history
    ]
    queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    tenant = principal.tenant

    # Guardrails before any model: injection, then scope, then PII redaction.
    from app.graph import audit
    from app.platform.guardrails import guard_chat_question

    guard = guard_chat_question(body.question, tenant)
    if not guard.allowed:
        message = (
            "I can't act on instructions embedded in a question. Ask about this "
            "customer's statement or the bank's products."
            if guard.reason == "injection"
            else "That is outside what I can see. I can answer questions about this "
            "customer's statement and this bank's products."
        )
        await audit.record(
            principal.tenant_id,
            statement_id=statement_id,
            node="guardrail.chat",
            event="blocked" if guard.reason == "injection" else "abstained",
            actor=principal.email,
            payload={"reason": guard.reason, **guard.detail},
        )
        logger.info("chat %s (%s)", guard.reason, guard.detail)

        async def refused() -> AsyncIterator[str]:
            yield f"event: {'blocked' if guard.reason == 'injection' else 'abstained'}\n"
            yield f"data: {json.dumps({'reason': guard.reason, **guard.detail})}\n\n"
            yield f"event: token\ndata: {json.dumps(message)}\n\n"
            yield f"event: done\ndata: {json.dumps(message)}\n\n"

        return StreamingResponse(
            refused(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    question = guard.question

    def produce() -> None:
        from langchain_community.callbacks.manager import get_openai_callback

        from app.core.config import settings
        from app.pipeline.chat import run_chat_turn
        from app.platform.tracing import (
            ATTR_STATEMENT_ID,
            ATTR_TENANT,
            ATTR_TENANT_ID,
            set_llm_usage,
            span,
        )

        try:
            with tenant_scope(tenant, user=principal.email):
                with span(
                    "llm.chat_turn",
                    **{
                        ATTR_TENANT: tenant,
                        ATTR_TENANT_ID: str(principal.tenant_id),
                        ATTR_STATEMENT_ID: str(statement_id),
                        "chat.history_turns": len(history),
                    },
                ):
                    with get_openai_callback() as cb:
                        for chunk in run_chat_turn(
                            question, history, metrics, df, tenant=tenant
                        ):
                            loop.call_soon_threadsafe(
                                queue.put_nowait, ("token", chunk)
                            )
                    set_llm_usage(
                        settings.openai_model, cb.prompt_tokens, cb.completion_tokens
                    )
            final = history[-1].content if history else ""
            loop.call_soon_threadsafe(queue.put_nowait, ("done", final))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
            logger.exception("chat failed")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        finally:
            await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
