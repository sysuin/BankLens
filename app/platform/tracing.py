"""
Tracing: every step, tool, token and dollar as an OpenTelemetry span.

Two exporters run side by side:

    PostgresSpanExporter   always on. Writes spans that carry a tenant id to
                           the `spans` table, so the console and the CLI can
                           show a trace with no external service at all.
    OTLP (HTTP)            when OTEL_EXPORTER_OTLP_ENDPOINT is set. Ships the
                           same spans to Jaeger / Tempo / a collector. The
                           Compose stack runs Jaeger on port 4318.

Span attributes carry ids and numbers only, never statement text. LLM spans
carry `llm.model`, `llm.tokens.in`, `llm.tokens.out`, `llm.cost_usd` and
`llm.prompt_version`, which is what makes "where did the time and the money
go" a query rather than an estimate.

Use `span(name, **attrs)` as a context manager in sync or async code, and
`set_llm_usage(model, tokens_in, tokens_out)` inside a span once a call has
returned. Context propagates through `asyncio.to_thread` because OpenTelemetry
rides on contextvars, so a node's spans nest under the run's span even when
the pipeline runs in a worker thread.
"""

from __future__ import annotations

import json
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from app.core.config import settings
from app.core.logger import get_logger
from app.platform.pricing import cost_usd

logger = get_logger(__name__)

ATTR_TENANT_ID = "banklens.tenant_id"
ATTR_TENANT = "banklens.tenant"
ATTR_RUN_ID = "banklens.run_id"
ATTR_STATEMENT_ID = "banklens.statement_id"
ATTR_REQUEST_ID = "banklens.request_id"
ATTR_USER = "banklens.user"

_configured = False
_provider: TracerProvider | None = None
_lock = threading.Lock()


# ── Postgres exporter ────────────────────────────────────────────────────────


class PostgresSpanExporter(SpanExporter):
    """
    Synchronous exporter used from the SDK's processor thread.

    Connects as the API role, not the owner. A batch can hold spans for
    several banks, so rows are grouped by tenant and each group is written
    in its own transaction with `app.tenant_id` pinned: the spans policy then
    refuses any row whose tenant does not match. Spans without a tenant
    attribute (e.g. /health) are not stored.
    """

    def __init__(self) -> None:
        self._conn = None
        self._conn_lock = threading.Lock()

    def _connection(self):
        import psycopg

        from app.db.session import _resolve_urls

        if self._conn is None or self._conn.closed:
            app_url, _ = _resolve_urls()
            conninfo = app_url.replace("postgresql+asyncpg://", "postgresql://", 1)
            self._conn = psycopg.connect(conninfo, autocommit=True)
        return self._conn

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = [row for row in (_span_row(s) for s in spans) if row is not None]
        if not rows:
            return SpanExportResult.SUCCESS
        try:
            by_tenant: dict[str, list[dict]] = {}
            for row in rows:
                by_tenant.setdefault(str(row["tenant_id"]), []).append(row)
            with self._conn_lock:
                conn = self._connection()
                for tenant_id, tenant_rows in by_tenant.items():
                    with conn.transaction(), conn.cursor() as cur:
                        cur.execute(
                            "SELECT set_config('app.tenant_id', %s, true)", (tenant_id,)
                        )
                        cur.executemany(
                            """
                        INSERT INTO spans (tenant_id, trace_id, span_id, parent_span_id, name,
                            kind, status, start_time, end_time, duration_ms, run_id,
                            statement_id, request_id, model, tokens_in, tokens_out,
                            cost_usd, attributes)
                        VALUES (%(tenant_id)s, %(trace_id)s, %(span_id)s, %(parent_span_id)s,
                            %(name)s, %(kind)s, %(status)s, %(start_time)s, %(end_time)s,
                            %(duration_ms)s, %(run_id)s, %(statement_id)s, %(request_id)s,
                            %(model)s, %(tokens_in)s, %(tokens_out)s, %(cost_usd)s,
                            %(attributes)s)
                        """,
                            tenant_rows,
                        )
            return SpanExportResult.SUCCESS
        except Exception as exc:  # noqa: BLE001 - tracing must never take the app down
            logger.warning("span export to Postgres failed: %s", exc)
            try:
                if self._conn is not None:
                    self._conn.close()
            finally:
                self._conn = None
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _ts(nanos: int | None) -> datetime:
    return datetime.fromtimestamp((nanos or 0) / 1e9, tz=timezone.utc)


def _uuid_or_none(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value else None
    except ValueError:
        return None


def _span_row(span: ReadableSpan) -> dict | None:
    attrs = dict(span.attributes or {})
    tenant_id = _uuid_or_none(attrs.get(ATTR_TENANT_ID))
    if tenant_id is None:
        return None
    ctx = span.get_span_context()
    parent = span.parent
    duration_ms = ((span.end_time or 0) - (span.start_time or 0)) / 1e6
    tokens_in = attrs.get("llm.tokens.in")
    tokens_out = attrs.get("llm.tokens.out")
    return {
        "tenant_id": tenant_id,
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "name": span.name[:128],
        "kind": span.kind.name.lower(),
        "status": span.status.status_code.name.lower(),
        "start_time": _ts(span.start_time),
        "end_time": _ts(span.end_time),
        "duration_ms": round(duration_ms, 3),
        "run_id": _uuid_or_none(attrs.get(ATTR_RUN_ID)),
        "statement_id": _uuid_or_none(attrs.get(ATTR_STATEMENT_ID)),
        "request_id": attrs.get(ATTR_REQUEST_ID),
        "model": attrs.get("llm.model"),
        "tokens_in": int(tokens_in) if tokens_in is not None else None,
        "tokens_out": int(tokens_out) if tokens_out is not None else None,
        "cost_usd": attrs.get("llm.cost_usd"),
        "attributes": json.dumps(attrs, default=str),
    }


# ── Setup ────────────────────────────────────────────────────────────────────


def setup_tracing() -> None:
    """Install the tracer provider once per process. Safe to call repeatedly."""
    global _configured, _provider
    with _lock:
        if _configured or not settings.tracing_enabled:
            return
        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(PostgresSpanExporter()))
        if settings.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            logger.info("OTLP trace export to %s", endpoint)
        trace.set_tracer_provider(provider)
        _provider = provider
        _configured = True
        logger.info("Tracing enabled (service=%s)", settings.otel_service_name)


def shutdown_tracing() -> None:
    global _configured, _provider
    with _lock:
        if _provider is not None:
            try:
                _provider.force_flush()
                _provider.shutdown()
            except Exception:  # noqa: BLE001
                pass
        _provider = None
        _configured = False


def tracer():
    return trace.get_tracer("banklens")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _context_attributes() -> dict[str, Any]:
    """Tenant / user / request id from the request context, when in scope."""
    from app.core import context

    attrs: dict[str, Any] = {}
    tenant = context._tenant.get()
    if tenant:
        attrs[ATTR_TENANT] = tenant
    user = context.current_user()
    if user:
        attrs[ATTR_USER] = user
    request_id = context.current_request_id()
    if request_id:
        attrs[ATTR_REQUEST_ID] = request_id
    return attrs


_INHERITED = (
    ATTR_TENANT_ID,
    ATTR_TENANT,
    ATTR_RUN_ID,
    ATTR_STATEMENT_ID,
    ATTR_REQUEST_ID,
    ATTR_USER,
)


def inherited_attributes(**attributes: Any) -> dict[str, Any]:
    """
    Attributes a new span should start with: the request context, then the
    given ones, then whatever identity the parent span carries that is
    still missing. Used by span() and by the gateway's per-call spans.
    """
    attrs = {
        **_context_attributes(),
        **{k: v for k, v in attributes.items() if v is not None},
    }
    parent = trace.get_current_span()
    parent_attrs = getattr(parent, "attributes", None) or {}
    for key in _INHERITED:
        if key not in attrs and key in parent_attrs:
            attrs[key] = parent_attrs[key]
    return attrs


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[trace.Span]:
    """
    Open a span, inheriting tenant/user/request attributes from the parent
    span when they are not given explicitly. Exceptions mark the span as an
    error and are re-raised.
    """
    attrs = inherited_attributes(**attributes)
    with tracer().start_as_current_span(name, attributes=attrs) as current:
        try:
            yield current
        except Exception as exc:
            current.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)[:200]))
            current.record_exception(exc)
            raise


def set_llm_usage(
    model: str, tokens_in: int, tokens_out: int, *, prompt_version: str | None = None
) -> float:
    """Attach model, tokens and cost to the current span. Returns the cost."""
    current = trace.get_current_span()
    cost = cost_usd(model, tokens_in, tokens_out)
    current.set_attribute("llm.provider", "openai")
    current.set_attribute("llm.model", model)
    current.set_attribute("llm.tokens.in", int(tokens_in))
    current.set_attribute("llm.tokens.out", int(tokens_out))
    current.set_attribute("llm.cost_usd", cost)
    if prompt_version:
        current.set_attribute("llm.prompt_version", prompt_version)
    return cost


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return f"{ctx.trace_id:032x}" if ctx and ctx.trace_id else None
