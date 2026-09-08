"""
BankLens API.

    uvicorn app.api.app:app --reload --port 8000

One process serves every tenant. Isolation comes from two places that agree:
the token names the tenant, and every database transaction pins that tenant
for row-level security. The middleware here gives each request an id and
clears the request context afterwards so nothing leaks between requests on
the same worker.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import (
    auth,
    customers,
    health,
    jobs,
    reviews,
    statements,
    traces,
    warehouse,
)
from app.api.security import assert_secret_is_safe_for
from app.core import context
from app.core.config import settings
from app.core.logger import get_logger
from app.db.session import dispose_engines
from app.platform.tracing import ATTR_REQUEST_ID, setup_tracing, shutdown_tracing, span

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    assert_secret_is_safe_for(settings.banklens_env)
    setup_tracing()
    from app.platform import gateway
    from app.warehouse.semantic import load_layer

    gateway.enable_budgets()
    # Fail fast: an unsafe template must stop the process, not a question.
    load_layer()
    logger.info(
        "BankLens API starting env=%s default_tenant=%s",
        settings.banklens_env,
        settings.default_tenant,
    )
    yield
    from app.graph.builder import close_checkpointer

    try:
        await close_checkpointer()
    finally:
        await dispose_engines()
        shutdown_tracing()
    logger.info("BankLens API stopped")


def create_app() -> FastAPI:
    application = FastAPI(
        title="BankLens Platform API",
        version=health.VERSION,
        description=(
            "Bank statement analysis for relationship managers: deterministic "
            "metrics, grounded product recommendations, tenant-isolated."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or context.new_request_id()
        tokens = context.set_request_context(
            tenant=None, user=None, request_id=request_id
        )
        started = time.perf_counter()
        try:
            # The request span. Streaming endpoints (run, review, chat) open
            # their own longer-lived spans that carry the tenant; this one
            # records the HTTP envelope.
            with span(
                f"http {request.method} {request.url.path}",
                **{ATTR_REQUEST_ID: request_id},
                **{"http.method": request.method, "http.path": request.url.path},
            ) as current:
                response = await call_next(request)
                current.set_attribute("http.status_code", response.status_code)
                principal = getattr(request.state, "principal", None)
                if principal is not None:
                    current.set_attribute("banklens.tenant", principal.tenant)
                    current.set_attribute(
                        "banklens.tenant_id", str(principal.tenant_id)
                    )
                    current.set_attribute("banklens.user", principal.email)
        except Exception:
            logger.exception("Unhandled error %s %s", request.method, request.url.path)
            response = JSONResponse(
                {"detail": "internal error", "request_id": request_id}, 500
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            # The endpoint ran in its own task, so its contextvars did not
            # propagate back here; the principal is on request.state instead.
            principal = getattr(request.state, "principal", None)
            logger.info(
                "%s %s -> %s in %.0f ms tenant=%s user=%s",
                request.method,
                request.url.path,
                getattr(response, "status_code", "?"),
                elapsed_ms,
                getattr(principal, "tenant", "-"),
                getattr(principal, "email", "-"),
            )
            context.reset_request_context(tokens)
        response.headers["x-request-id"] = request_id
        return response

    application.include_router(health.router)
    application.include_router(auth.router)
    application.include_router(customers.router)
    application.include_router(statements.router)
    application.include_router(reviews.router)
    application.include_router(traces.router)
    application.include_router(jobs.router)
    application.include_router(warehouse.router)
    return application


app = create_app()
